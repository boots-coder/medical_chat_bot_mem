/**
 * External Medical System Test Page
 * 外部医疗系统模拟测试页面逻辑
 */

let currentSessionId = null;

/**
 * 填充测试数据
 */
function fillTestData() {
    const timestamp = Date.now();
    document.getElementById('patient_id').value = `P${timestamp}`;
    document.getElementById('patient_name').value = '张三';
    document.getElementById('patient_age').value = '45';
    document.getElementById('gender').value = 'male';
    document.getElementById('doctor_name').value = '李医生';
    document.getElementById('department').value = '心内科';
    document.getElementById('appointment_id').value = `APT${timestamp}`;
}

/**
 * 创建会话
 */
document.getElementById('create-session-form').addEventListener('submit', async function(e) {
    e.preventDefault();

    // 隐藏错误和摘要
    hideError();
    document.getElementById('session-summary').classList.add('hidden');

    // 获取表单数据
    const formData = {
        patient_id: document.getElementById('patient_id').value,
        patient_name: document.getElementById('patient_name').value,
        patient_age: parseInt(document.getElementById('patient_age').value),
        gender: document.getElementById('gender').value,
        doctor_name: document.getElementById('doctor_name').value,
        department: document.getElementById('department').value,
        appointment_id: document.getElementById('appointment_id').value
    };

    console.log('正在创建会话...', formData);

    try {
        const response = await fetch('/api/external/create-session', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '创建会话失败');
        }

        const data = await response.json();
        console.log('会话创建成功:', data);

        // 保存会话ID
        currentSessionId = data.session_id;

        // 显示会话信息
        displaySessionInfo(data);

    } catch (error) {
        console.error('创建会话失败:', error);
        showError(error.message);
    }
});

/**
 * 显示会话信息
 */
function displaySessionInfo(data) {
    document.getElementById('display-session-id').value = data.session_id;
    document.getElementById('display-url').value = data.url;
    document.getElementById('display-token').value = data.url_token;
    document.getElementById('display-expires').value = formatDateTime(data.expires_at);

    // 设置打开聊天链接
    document.getElementById('open-chat-link').href = data.url;

    // 显示会话信息区域
    document.getElementById('session-info').classList.remove('hidden');

    // 隐藏表单
    document.getElementById('create-session-form').parentElement.classList.add('hidden');
}

/**
 * 复制URL到剪贴板
 */
function copyURL() {
    const urlInput = document.getElementById('display-url');
    urlInput.select();
    urlInput.setSelectionRange(0, 99999); // 移动设备兼容

    try {
        document.execCommand('copy');
        alert('URL已复制到剪贴板！');
    } catch (error) {
        console.error('复制失败:', error);
        alert('复制失败，请手动复制');
    }
}

/**
 * 获取会话摘要
 */
async function getSummary() {
    if (!currentSessionId) {
        showError('没有可用的会话ID');
        return;
    }

    hideError();

    try {
        const response = await fetch(`/api/session/${currentSessionId}/summary`);

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '获取摘要失败');
        }

        const summary = await response.json();
        console.log('会话摘要:', summary);

        // 显示摘要
        document.getElementById('summary-content').textContent = JSON.stringify(summary, null, 2);
        document.getElementById('session-summary').classList.remove('hidden');

    } catch (error) {
        console.error('获取摘要失败:', error);
        showError(error.message);
    }
}

/**
 * 获取长期记忆摘要
 */
async function getMemorySummary() {
    if (!currentSessionId) {
        showError('没有可用的会话ID');
        return;
    }

    hideError();

    try {
        const response = await fetch(`/api/session/${currentSessionId}/memory-summary`);

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '获取长期记忆摘要失败');
        }

        const memory = await response.json();
        console.log('长期记忆摘要:', memory);

        // 显示格式化的摘要
        document.getElementById('mem-topic').textContent = memory.session_topic || '无';
        document.getElementById('mem-narrative').textContent = memory.narrative_summary || '无';
        document.getElementById('mem-complaint').textContent = memory.main_complaint || '无';
        document.getElementById('mem-rounds').textContent = memory.dialogue_rounds || 0;
        document.getElementById('mem-entities').textContent = memory.knowledge_graph.entities_count || 0;
        document.getElementById('mem-relations').textContent = memory.knowledge_graph.relationships_count || 0;

        document.getElementById('memory-summary').classList.remove('hidden');

    } catch (error) {
        console.error('获取长期记忆摘要失败:', error);
        showError(error.message);
    }
}

/**
 * 重置表单
 */
function resetForm() {
    // 重置所有字段
    document.getElementById('create-session-form').reset();
    currentSessionId = null;

    // 显示表单，隐藏会话信息和摘要
    document.getElementById('create-session-form').parentElement.classList.remove('hidden');
    document.getElementById('session-info').classList.add('hidden');
    document.getElementById('session-summary').classList.add('hidden');
    document.getElementById('memory-summary').classList.add('hidden');

    hideError();
}

/**
 * 显示错误信息
 */
function showError(message) {
    document.getElementById('error-text').textContent = message;
    document.getElementById('error-message').classList.remove('hidden');

    // 滚动到错误消息
    document.getElementById('error-message').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * 隐藏错误信息
 */
function hideError() {
    document.getElementById('error-message').classList.add('hidden');
}

/**
 * 格式化日期时间
 */
function formatDateTime(isoString) {
    const date = new Date(isoString);
    return date.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

/**
 * 页面加载时初始化
 */
window.addEventListener('DOMContentLoaded', () => {
    console.log('外部医疗系统测试页面已加载');

    // 自动填充一些测试数据（仅用于开发）
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        console.log('检测到本地环境，可以使用"填充测试数据"按钮快速测试');
    }
});
