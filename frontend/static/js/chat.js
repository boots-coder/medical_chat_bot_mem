/**
 * Medical Chat WebSocket Client
 * Medical consultation chat interface WebSocket client logic
 */

let ws = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;

/**
 * Initialize WebSocket connection
 */
function connectWebSocket() {
    const wsUrl = `ws://${window.location.hostname}:${wsPort}/ws/${sessionId}`;

    console.log('Connecting to WebSocket:', wsUrl);
    updateConnectionStatus('connecting', 'Connecting...');

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket connection established');
        reconnectAttempts = 0;
        updateConnectionStatus('connected', 'Connected');
        addSystemMessage('Connection successful, you can start consulting');
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            console.log('Received message:', data);

            if (data.type === 'assistant') {
                addMessage('assistant', data.content);
            } else if (data.type === 'error') {
                addSystemMessage('Error: ' + data.content, 'error');
            }
        } catch (error) {
            console.error('Failed to parse message:', error);
            addSystemMessage('Received invalid message', 'error');
        }
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateConnectionStatus('error', 'Connection error');
        addSystemMessage('Connection error, please refresh the page', 'error');
    };

    ws.onclose = () => {
        console.log('WebSocket connection closed');
        updateConnectionStatus('disconnected', 'Connection closed');

        // Attempt to reconnect
        if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
            reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 10000);
            console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts})`);

            setTimeout(() => {
                addSystemMessage(`Attempting to reconnect... (${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})`, 'warning');
                connectWebSocket();
            }, delay);
        } else {
            addSystemMessage('Unable to connect to server, please refresh the page and try again', 'error');
        }
    };
}

/**
 * Send message
 */
function sendMessage() {
    const input = document.getElementById('message-input');
    const message = input.value.trim();

    if (!message) {
        return;
    }

    if (!ws || ws.readyState !== WebSocket.OPEN) {
        addSystemMessage('Connection not ready, please try again later', 'error');
        return;
    }

    // Display user message
    addMessage('user', message);

    // Send to server
    try {
        ws.send(JSON.stringify({
            type: 'user',
            content: message
        }));

        // Clear input field
        input.value = '';
    } catch (error) {
        console.error('Failed to send message:', error);
        addSystemMessage('Failed to send, please try again', 'error');
    }
}

/**
 * Add message to chat area
 * @param {string} role - 'user' or 'assistant'
 * @param {string} content - Message content
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

    // Handle line breaks
    bubble.innerHTML = content.replace(/\n/g, '<br>');

    div.appendChild(bubble);
    container.appendChild(div);

    // Scroll to bottom
    container.scrollTop = container.scrollHeight;
}

/**
 * Add system message
 * @param {string} content - Message content
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
 * Update connection status
 * @param {string} status - 'connecting', 'connected', 'disconnected', 'error'
 * @param {string} text - Status text
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
 * End session
 */
async function endSession() {
    if (!confirm('Are you sure you want to end the consultation? You will not be able to continue the conversation after ending.')) {
        return;
    }

    try {
        const response = await fetch(`/api/session/${sessionId}/end`, {
            method: 'POST'
        });

        if (response.ok) {
            addSystemMessage('Session ended, thank you for using our service');
            if (ws) {
                ws.close();
            }
            document.getElementById('message-input').disabled = true;

            // Disable send and end buttons
            document.querySelectorAll('button').forEach(btn => {
                btn.disabled = true;
                btn.classList.add('opacity-50', 'cursor-not-allowed');
            });
        } else {
            const error = await response.json();
            addSystemMessage('Failed to end session: ' + (error.detail || 'Unknown error'), 'error');
        }
    } catch (error) {
        console.error('Failed to end session:', error);
        addSystemMessage('Failed to end session: ' + error.message, 'error');
    }
}

/**
 * Initialize on page load
 */
window.addEventListener('DOMContentLoaded', () => {
    console.log('Page loaded, initializing chat interface');
    console.log('Session ID:', sessionId);

    // Connect WebSocket
    connectWebSocket();

    // Auto-focus input field
    document.getElementById('message-input').focus();
});

// Close WebSocket on page unload
window.addEventListener('beforeunload', () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
    }
});
