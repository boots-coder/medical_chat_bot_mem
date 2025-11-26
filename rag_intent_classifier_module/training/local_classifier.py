"""
本地RAG意图分类器
基于轻量级预训练模型（DistilBERT/RoBERTa）进行微调
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import json
from pathlib import Path
from tqdm import tqdm


class RAGIntentDataset(Dataset):
    """RAG意图分类数据集"""

    def __init__(
        self,
        samples: List[Dict],
        tokenizer,
        max_length: int = 256
    ):
        """
        Args:
            samples: 样本列表，每个样本包含 query, short_term_context, need_rag
            tokenizer: 预训练模型的tokenizer
            max_length: 最大序列长度
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 拼接query和context
        query = sample['query']
        context = sample.get('short_term_context', '')

        # 格式: [CLS] query [SEP] context [SEP]
        if context:
            text = f"{query} [SEP] {context}"
        else:
            text = query

        # Tokenization
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 标签: need_rag -> 1, 不需要 -> 0
        label = 1 if sample['need_rag'] else 0

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


class LocalRAGIntentClassifier:
    """本地RAG意图分类器"""

    def __init__(
        self,
        model_name: str = "hfl/chinese-roberta-wwm-ext",
        device: str = None
    ):
        """
        初始化分类器

        Args:
            model_name: 预训练模型名称
                - "hfl/chinese-roberta-wwm-ext": 中文RoBERTa（推荐）
                - "bert-base-chinese": BERT中文
                - "distilbert-base-multilingual-cased": 多语言DistilBERT
            device: 设备 (cuda/cpu)
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"初始化本地RAG意图分类器...")
        print(f"  模型: {model_name}")
        print(f"  设备: {self.device}")

        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2  # 二分类: need_rag / no_need_rag
        )
        self.model.to(self.device)

        print(f"✓ 模型加载成功")

    def train(
        self,
        train_samples: List[Dict],
        eval_samples: List[Dict] = None,
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        output_dir: str = "./models/checkpoints"
    ) -> Dict[str, List[float]]:
        """
        训练模型

        Args:
            train_samples: 训练样本列表
            eval_samples: 验证样本列表（可选）
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            warmup_ratio: warmup比例
            output_dir: 模型保存目录

        Returns:
            训练历史 {train_loss, eval_loss, eval_accuracy, eval_f1}
        """
        print("\n" + "="*60)
        print("开始训练")
        print("="*60)

        # 创建数据集
        train_dataset = RAGIntentDataset(train_samples, self.tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        eval_loader = None
        if eval_samples:
            eval_dataset = RAGIntentDataset(eval_samples, self.tokenizer)
            eval_loader = DataLoader(eval_dataset, batch_size=batch_size)

        # 优化器和学习率调度
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # 训练历史
        history = {
            'train_loss': [],
            'eval_loss': [],
            'eval_accuracy': [],
            'eval_f1': []
        }

        # 训练循环
        best_f1 = 0.0
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # 训练阶段
            self.model.train()
            train_loss = 0.0

            progress_bar = tqdm(train_loader, desc="Training")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            print(f"  训练损失: {avg_train_loss:.4f}")

            # 验证阶段
            if eval_loader:
                eval_metrics = self.evaluate(eval_loader)
                history['eval_loss'].append(eval_metrics['loss'])
                history['eval_accuracy'].append(eval_metrics['accuracy'])
                history['eval_f1'].append(eval_metrics['f1'])

                print(f"  验证损失: {eval_metrics['loss']:.4f}")
                print(f"  验证准确率: {eval_metrics['accuracy']:.4f}")
                print(f"  验证F1: {eval_metrics['f1']:.4f}")

                # 保存最佳模型
                if eval_metrics['f1'] > best_f1:
                    best_f1 = eval_metrics['f1']
                    self.save_model(output_dir)
                    print(f"  ✓ 保存最佳模型 (F1={best_f1:.4f})")

        print("\n" + "="*60)
        print("训练完成")
        print(f"最佳F1分数: {best_f1:.4f}")
        print("="*60)

        return history

    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """
        评估模型

        Returns:
            {loss, accuracy, precision, recall, f1}
        """
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs.loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(eval_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def predict(
        self,
        query: str,
        short_term_context: str = "",
        return_confidence: bool = True
    ) -> Dict:
        """
        预测单个样本

        Args:
            query: 用户查询
            short_term_context: 短期记忆上下文
            return_confidence: 是否返回置信度

        Returns:
            {need_rag: bool, confidence: float}
        """
        self.model.eval()

        # 拼接文本
        if short_term_context:
            text = f"{query} [SEP] {short_term_context}"
        else:
            text = query

        # Tokenization
        encoding = self.tokenizer(
            text,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

        pred = torch.argmax(logits, dim=1).item()
        confidence = probs[0][pred].item()

        result = {
            'need_rag': bool(pred),
            'confidence': confidence
        }

        return result

    def save_model(self, output_dir: str):
        """保存模型"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        # 保存自定义配置（不覆盖模型的config.json）
        custom_config = {
            'model_name': self.model_name,
            'device': self.device
        }
        with open(output_path / 'training_config.json', 'w', encoding='utf-8') as f:
            json.dump(custom_config, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_model(cls, model_dir: str, device: str = None):
        """加载已保存的模型"""
        model_path = Path(model_dir)

        # 加载自定义配置
        custom_config_path = model_path / 'training_config.json'
        if custom_config_path.exists():
            with open(custom_config_path, 'r', encoding='utf-8') as f:
                custom_config = json.load(f)
            model_name = custom_config['model_name']
        else:
            # 兼容旧版本，默认使用xlm-roberta-base
            model_name = "xlm-roberta-base"

        # 直接从目录加载模型和tokenizer
        device = device if device else ('cuda' if torch.cuda.is_available() else
                                       ('mps' if torch.backends.mps.is_available() else 'cpu'))

        classifier = cls.__new__(cls)
        classifier.model_name = model_name
        classifier.device = device

        # 加载模型和tokenizer
        classifier.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        classifier.model.to(classifier.device)
        classifier.tokenizer = AutoTokenizer.from_pretrained(model_path)

        print(f"✓ 模型加载成功: {model_dir}")

        return classifier


if __name__ == "__main__":
    print("本地RAG意图分类器 - 测试")

    # 创建分类器
    classifier = LocalRAGIntentClassifier(
        model_name="hfl/chinese-roberta-wwm-ext"
    )

    # 测试样本
    test_samples = [
        {
            "query": "上次医生开的降压药我能继续吃吗?",
            "short_term_context": "",
            "need_rag": True
        },
        {
            "query": "我现在头痛怎么办?",
            "short_term_context": "",
            "need_rag": False
        }
    ]

    print("\n测试预测功能（未训练模型）:")
    for sample in test_samples:
        result = classifier.predict(
            query=sample['query'],
            short_term_context=sample['short_term_context']
        )
        print(f"\nQuery: {sample['query']}")
        print(f"预测: need_rag={result['need_rag']}, confidence={result['confidence']:.3f}")
        print(f"真实: need_rag={sample['need_rag']}")
