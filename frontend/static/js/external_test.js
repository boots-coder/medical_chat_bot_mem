/**
 * External Medical System Test Page
 * Test page logic for external medical system simulation
 */

let currentSessionId = null;

/**
 * Fill test data
 */
function fillTestData() {
    const timestamp = Date.now();
    document.getElementById('patient_id').value = `P${timestamp}`;
    document.getElementById('patient_name').value = 'John Doe';
    document.getElementById('patient_age').value = '45';
    document.getElementById('gender').value = 'male';
    document.getElementById('doctor_name').value = 'Dr. Smith';
    document.getElementById('department').value = 'Cardiology';
    document.getElementById('appointment_id').value = `APT${timestamp}`;
}

/**
 * Create session
 */
document.getElementById('create-session-form').addEventListener('submit', async function(e) {
    e.preventDefault();

    // Hide error and summary
    hideError();
    document.getElementById('session-summary').classList.add('hidden');

    // Get form data
    const formData = {
        patient_id: document.getElementById('patient_id').value,
        patient_name: document.getElementById('patient_name').value,
        patient_age: parseInt(document.getElementById('patient_age').value),
        gender: document.getElementById('gender').value,
        doctor_name: document.getElementById('doctor_name').value,
        department: document.getElementById('department').value,
        appointment_id: document.getElementById('appointment_id').value
    };

    console.log('Creating session...', formData);

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
            throw new Error(error.detail || 'Failed to create session');
        }

        const data = await response.json();
        console.log('Session created successfully:', data);

        // Save session ID
        currentSessionId = data.session_id;

        // Display session info
        displaySessionInfo(data);

    } catch (error) {
        console.error('Failed to create session:', error);
        showError(error.message);
    }
});

/**
 * Display session info
 */
function displaySessionInfo(data) {
    document.getElementById('display-session-id').value = data.session_id;
    document.getElementById('display-url').value = data.url;
    document.getElementById('display-token').value = data.url_token;
    document.getElementById('display-expires').value = formatDateTime(data.expires_at);

    // Set open chat link
    document.getElementById('open-chat-link').href = data.url;

    // Show session info area
    document.getElementById('session-info').classList.remove('hidden');

    // Hide form
    document.getElementById('create-session-form').parentElement.classList.add('hidden');
}

/**
 * Copy URL to clipboard
 */
function copyURL() {
    const urlInput = document.getElementById('display-url');
    urlInput.select();
    urlInput.setSelectionRange(0, 99999); // Mobile device compatibility

    try {
        document.execCommand('copy');
        alert('URL copied to clipboard!');
    } catch (error) {
        console.error('Copy failed:', error);
        alert('Copy failed, please copy manually');
    }
}

/**
 * Get session summary
 */
async function getSummary() {
    if (!currentSessionId) {
        showError('No session ID available');
        return;
    }

    hideError();

    try {
        const response = await fetch(`/api/session/${currentSessionId}/summary`);

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to get summary');
        }

        const summary = await response.json();
        console.log('Session summary:', summary);

        // Display summary
        document.getElementById('summary-content').textContent = JSON.stringify(summary, null, 2);
        document.getElementById('session-summary').classList.remove('hidden');

    } catch (error) {
        console.error('Failed to get summary:', error);
        showError(error.message);
    }
}

/**
 * Get long-term memory summary
 */
async function getMemorySummary() {
    if (!currentSessionId) {
        showError('No session ID available');
        return;
    }

    hideError();

    try {
        const response = await fetch(`/api/session/${currentSessionId}/memory-summary`);

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to get long-term memory summary');
        }

        const memory = await response.json();
        console.log('Long-term memory summary:', memory);

        // Display formatted summary
        document.getElementById('mem-topic').textContent = memory.session_topic || 'None';
        document.getElementById('mem-narrative').textContent = memory.narrative_summary || 'None';
        document.getElementById('mem-complaint').textContent = memory.main_complaint || 'None';
        document.getElementById('mem-rounds').textContent = memory.dialogue_rounds || 0;
        document.getElementById('mem-entities').textContent = memory.knowledge_graph.entities_count || 0;
        document.getElementById('mem-relations').textContent = memory.knowledge_graph.relationships_count || 0;

        document.getElementById('memory-summary').classList.remove('hidden');

    } catch (error) {
        console.error('Failed to get long-term memory summary:', error);
        showError(error.message);
    }
}

/**
 * Reset form
 */
function resetForm() {
    // Reset all fields
    document.getElementById('create-session-form').reset();
    currentSessionId = null;

    // Show form, hide session info and summary
    document.getElementById('create-session-form').parentElement.classList.remove('hidden');
    document.getElementById('session-info').classList.add('hidden');
    document.getElementById('session-summary').classList.add('hidden');
    document.getElementById('memory-summary').classList.add('hidden');

    hideError();
}

/**
 * Show error message
 */
function showError(message) {
    document.getElementById('error-text').textContent = message;
    document.getElementById('error-message').classList.remove('hidden');

    // Scroll to error message
    document.getElementById('error-message').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Hide error message
 */
function hideError() {
    document.getElementById('error-message').classList.add('hidden');
}

/**
 * Format date and time
 */
function formatDateTime(isoString) {
    const date = new Date(isoString);
    return date.toLocaleString('en-US', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

/**
 * Initialize on page load
 */
window.addEventListener('DOMContentLoaded', () => {
    console.log('External medical system test page loaded');

    // Auto-fill some test data (for development only)
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        console.log('Local environment detected, you can use "Fill Test Data" button for quick testing');
    }
});
