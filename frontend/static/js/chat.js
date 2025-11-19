/**
 * Medical Chat WebSocket Client
 * 医疗咨询聊天界面 WebSocket 客户端逻辑
 */

let ws = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;

/**
 * 初始化 WebSocket 连接
 */
function connectWebSocket() {
    const wsUrl = `ws://${window.location.hostname}:${wsPort}/ws/${sessionId}`;

    console.log('正在连接 WebSocket:', wsUrl);
    updateConnectionStatus('connecting', '正在连接...');

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket 连接已建立');
        reconnectAttempts = 0;
        updateConnectionStatus('connected', '已连接');
        addSystemMessage('连接成功，可以开始咨询');
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            console.log('收到消息:', data);

            if (data.type === 'assistant') {
                addMessage('assistant', data.content);
            } else if (data.type === 'error') {
                addSystemMessage('错误: ' + data.content, 'error');
            }
        } catch (error) {
            console.error('解析消息失败:', error);
            addSystemMessage('收到无效消息', 'error');
        }
    };

    ws.onerror = (error) => {
        console.error('WebSocket 错误:', error);
        updateConnectionStatus('error', '连接错误');
        addSystemMessage('连接错误，请刷新页面', 'error');
    };

    ws.onclose = () => {
        console.log('WebSocket 连接已关闭');
        updateConnectionStatus('disconnected', '连接已断开');

        // 尝试重连
        if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
            reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 10000);
            console.log(`${delay}ms 后尝试重连 (第 ${reconnectAttempts} 次)`);

            setTimeout(() => {
                addSystemMessage(`尝试重新连接... (${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})`, 'warning');
                connectWebSocket();
            }, delay);
        } else {
            addSystemMessage('无法连接到服务器，请刷新页面重试', 'error');
        }
    };
}

/**
 * 发送消息
 */
function sendMessage() {
    const input = document.getElementById('message-input');
    const message = input.value.trim();

    if (!message) {
        return;
    }

    if (!ws || ws.readyState !== WebSocket.OPEN) {
        addSystemMessage('连接未就绪，请稍候重试', 'error');
        return;
    }

    // 显示用户消息
    addMessage('user', message);

    // 发送到服务器
    try {
        ws.send(JSON.stringify({
            type: 'user',
            content: message
        }));

        // 清空输入框
        input.value = '';
    } catch (error) {
        console.error('发送消息失败:', error);
        addSystemMessage('发送失败，请重试', 'error');
    }
}

/**
 * 添加消息到聊天区域
 * @param {string} role - 'user' 或 'assistant'
 * @param {string} content - 消息内容
 */
function addMessage(role, content) {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = `mb-3 ${role === 'user' ? 'text-right' : 'text-left'}`;

    const bubble = document.createElement('div');
    bubble.className = `inline-block px-4 py-2 rounded-lg max-w-xl ${
        role === 'user'
            ? 'bg-blue-500 text-white'
            : 'bg-gray-200 text-gray-800'
    }`;

    // 处理换行
    bubble.innerHTML = content.replace(/\n/g, '<br>');

    div.appendChild(bubble);
    container.appendChild(div);

    // 滚动到底部
    container.scrollTop = container.scrollHeight;
}

/**
 * 添加系统消息
 * @param {string} content - 消息内容
 * @param {string} type - 'info', 'warning', 'error'
 */
function addSystemMessage(content, type = 'info') {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');

    let colorClass = 'text-gray-500';
    if (type === 'error') colorClass = 'text-red-500';
    if (type === 'warning') colorClass = 'text-yellow-600';

    div.className = `text-center ${colorClass} text-sm my-2`;
    div.textContent = content;

    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

/**
 * 更新连接状态
 * @param {string} status - 'connecting', 'connected', 'disconnected', 'error'
 * @param {string} text - 状态文本
 */
function updateConnectionStatus(status, text) {
    const statusElement = document.getElementById('status-text');
    if (!statusElement) return;

    statusElement.textContent = text;

    const statusColors = {
        connecting: 'text-yellow-600',
        connected: 'text-green-600',
        disconnected: 'text-gray-500',
        error: 'text-red-600'
    };

    statusElement.className = statusColors[status] || 'text-gray-500';
}

/**
 * 结束会话
 */
async function endSession() {
    if (!confirm('确定要结束咨询吗？结束后将无法继续对话。')) {
        return;
    }

    try {
        const response = await fetch(`/api/session/${sessionId}/end`, {
            method: 'POST'
        });

        if (response.ok) {
            addSystemMessage('会话已结束，感谢您的使用');
            if (ws) {
                ws.close();
            }
            document.getElementById('message-input').disabled = true;

            // 禁用发送和结束按钮
            document.querySelectorAll('button').forEach(btn => {
                btn.disabled = true;
                btn.classList.add('opacity-50', 'cursor-not-allowed');
            });
        } else {
            const error = await response.json();
            addSystemMessage('结束会话失败: ' + (error.detail || '未知错误'), 'error');
        }
    } catch (error) {
        console.error('结束会话失败:', error);
        addSystemMessage('结束会话失败: ' + error.message, 'error');
    }
}

/**
 * 页面加载时初始化
 */
window.addEventListener('DOMContentLoaded', () => {
    console.log('页面加载完成，初始化聊天界面');
    console.log('Session ID:', sessionId);

    // 连接 WebSocket
    connectWebSocket();

    // 自动聚焦输入框
    document.getElementById('message-input').focus();
});

// 页面卸载时关闭WebSocket
window.addEventListener('beforeunload', () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
    }
});
