const BASE = ''; // Relative path

const BOT_AVATAR_SVG = `<svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <circle cx="12" cy="12" r="4" stroke-width="2"/>
    <path stroke-linecap="round" stroke-width="2" d="M12 2v3M12 19v3M2 12h3M19 12h3M4.93 4.93l2.12 2.12M16.95 16.95l2.12 2.12M4.93 19.07l2.12-2.12M16.95 7.05l2.12-2.12"/>
</svg>`;

function getThreadId() {
    let t = localStorage.getItem('chat_thread_id');
    if (!t) {
        t = (crypto && crypto.randomUUID) ? crypto.randomUUID().replace(/-/g, '').slice(0, 16)
            : (Date.now().toString(36) + Math.random().toString(36).slice(2, 10));
        localStorage.setItem('chat_thread_id', t);
    }
    return t;
}

function newConversation() {
    localStorage.removeItem('chat_thread_id');
    const history = document.getElementById('chat-history');
    history.querySelectorAll('.message').forEach(el => el.remove());
    const hero = document.getElementById('welcome-hero');
    if (hero) hero.style.display = '';
    document.getElementById('query-input').focus();
    if (window.innerWidth <= 768) closeSidebar();
}

document.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        newConversation();
    }
});

// Init
checkHealth();

function setView(view) {
    document.querySelectorAll('.view').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));

    document.getElementById('view-' + view).classList.add('active');
    document.getElementById('nav-' + view).classList.add('active');
    const mobileNav = document.getElementById('nav-' + view + '-mobile');
    if (mobileNav) mobileNav.classList.add('active');

    if (view === 'admin') refreshDocs();

    // Always close mobile sidebar after navigation to reveal content
    if (window.innerWidth <= 768) {
        closeSidebar();
    }
}

function setStatusValue(key, text) {
    const el = document.querySelector(`[data-status="${key}"]`);
    if (el) el.textContent = text;
}

async function checkHealth() {
    const headerDot = document.getElementById('header-health-dot');
    const headerText = document.getElementById('header-health-text');
    const sideDot = document.getElementById('health-dot');
    try {
        const res = await fetch(BASE + '/health');
        const data = await res.json();
        if (data.demo) {
            const banner = document.getElementById('demo-banner');
            if (banner) banner.style.display = 'flex';
        }
        const comps = data.components || {};
        const gemini = (comps.gemini || {}).status === 'healthy';
        const qdrant = (comps.qdrant || {}).status === 'healthy';
        setStatusValue('engine', gemini ? 'Gemini · live' : 'simulated');
        const points = (comps.qdrant || {}).points_count;
        setStatusValue('qdrant', qdrant ? ('Qdrant' + (points != null ? ` · ${points} pts` : '')) : 'in-memory');
        const healthy = data.status === 'healthy';
        setStatusValue('overall', healthy ? 'operational' : 'degraded');
        const cls = healthy ? 'status-dot ok' : 'status-dot warn';
        if (headerDot) headerDot.className = cls;
        if (sideDot) sideDot.className = cls;
        if (headerText) headerText.textContent = healthy ? 'All systems operational' : 'Degraded';
    } catch (e) {
        console.error("Health check failed", e);
        setStatusValue('overall', 'offline');
        if (headerDot) headerDot.className = 'status-dot err';
        if (sideDot) sideDot.className = 'status-dot err';
        if (headerText) headerText.textContent = 'Unreachable';
    }
}

// --- Chat Logic ---

function handleKeyPress(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        doSearch();
    }
}

// Auto-grow the composer textarea
const queryInput = document.getElementById('query-input');
queryInput.addEventListener('input', () => {
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(queryInput.scrollHeight, 200) + 'px';
});

// Chat attachments are image-only. Document ingestion lives in /backend.
let chatAttachments = [];
const IMAGE_EXTS = ['png', 'jpg', 'jpeg', 'webp', 'gif'];

function handleChatFileSelect(e) {
    const files = Array.from(e.target.files || []);
    const imageFiles = files.filter(f => IMAGE_EXTS.includes((f.name.split('.').pop() || '').toLowerCase()));
    if (imageFiles.length !== files.length) {
        showToast('Document ingestion has moved to the Operations Console. Images can still be attached here for analysis.', false);
    }
    chatAttachments = chatAttachments.concat(imageFiles);
    renderChatAttachments();
    e.target.value = '';
}

function renderChatAttachments() {
    const wrap = document.getElementById('chat-attachments');
    if (!wrap) return;
    if (chatAttachments.length === 0) {
        wrap.hidden = true;
        wrap.innerHTML = '';
        return;
    }
    wrap.hidden = false;
    wrap.replaceChildren(...chatAttachments.map((f, idx) => {
        const ext = (f.name.split('.').pop() || '').toLowerCase();
        const icon = IMAGE_EXTS.includes(ext) ? '🖼️' : '📄';
        const chip = document.createElement('span');
        chip.className = 'chat-attachment-chip';
        chip.append(`${icon} ${f.name} `);
        const btn = document.createElement('button');
        btn.title = 'Remove';
        btn.textContent = '×';
        btn.addEventListener('click', () => removeChatAttachment(idx));
        chip.appendChild(btn);
        return chip;
    }));
}

function removeChatAttachment(idx) {
    chatAttachments.splice(idx, 1);
    renderChatAttachments();
}

function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const result = reader.result || '';
            const comma = String(result).indexOf(',');
            resolve(comma >= 0 ? String(result).slice(comma + 1) : String(result));
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

function showTyping() {
    removeTyping();
    const history = document.getElementById('chat-history');
    const div = document.createElement('div');
    div.className = 'message bot';
    div.id = 'typing-message';
    div.innerHTML = `
        <div class="msg-avatar">${BOT_AVATAR_SVG}</div>
        <div class="msg-main">
            <div class="message-label">Helios <span class="label-tag">Assistant</span></div>
            <div class="typing-indicator" aria-label="Helios is thinking"><span></span><span></span><span></span></div>
        </div>`;
    history.appendChild(div);
    history.scrollTop = history.scrollHeight;
}

function removeTyping() {
    const el = document.getElementById('typing-message');
    if (el) el.remove();
}

async function doSearch() {
    const input = document.getElementById('query-input');
    const query = input.value.trim();
    const hasAttachments = chatAttachments.length > 0;
    if (!query && !hasAttachments) return;

    const userLabel = hasAttachments
        ? `${query || '(see attachment)'}  📎 ${chatAttachments.length} file(s)`
        : query;
    addMessage('user', userLabel);
    input.value = '';
    input.style.height = 'auto';

    const attached = chatAttachments.slice();
    chatAttachments = [];
    renderChatAttachments();

    const btn = document.getElementById('send-btn');
    const icon = document.getElementById('send-icon');
    const spinner = document.getElementById('send-spinner');
    btn.disabled = true;
    icon.style.display = 'none';
    spinner.style.display = 'inline-block';
    showTyping();

    try {
        const imagePayload = [];
        for (const f of attached) {
            const ext = (f.name.split('.').pop() || '').toLowerCase();
            if (IMAGE_EXTS.includes(ext)) {
                const b64 = await fileToBase64(f);
                imagePayload.push({ data_base64: b64, mime_type: f.type || `image/${ext === 'jpg' ? 'jpeg' : ext}` });
            }
        }

        const body = { query: query || 'Please analyze the attached content.', top_k: 10, thread_id: getThreadId() };
        if (imagePayload.length > 0) body.images = imagePayload;

        const headers = { 'Content-Type': 'application/json' };

        const res = await fetch(BASE + '/search', {
            method: 'POST',
            headers,
            body: JSON.stringify(body)
        });
        if (res.status === 401) {
            removeTyping();
            addMessage('bot', 'Søgning kræver et access token. Kontakt ejeren for adgang til den private demo.');
            return;
        }
        const data = await res.json().catch(() => ({}));
        removeTyping();
        if (!res.ok) {
            addMessage('bot', `Søgning fejlede (${res.status}): ${data.error || 'unknown error'}`);
            return;
        }
        const answer = data.answer || data.message || 'Her er de mest relevante resultater fra dine dokumenter:';
        addMessage('bot', answer, data.results, {
            total_ms: data.total_ms,
            ai_ms: data.ai_ms,
            sources: (data.sources || []).length
        });
    } catch (e) {
        removeTyping();
        addMessage('bot', 'Sorry, an error occurred while searching: ' + e.message);
    } finally {
        btn.disabled = false;
        icon.style.display = 'inline';
        spinner.style.display = 'none';
        document.getElementById('query-input').focus();
    }
}

function renderBubbleContent(type, text) {
    // User messages: escape only, keep simple. Bot messages: render Markdown safely.
    if (!text) return '';
    if (type === 'user') return escapeHtml(text);
    try {
        const html = marked.parse(String(text), { breaks: true, gfm: true });
        return DOMPurify.sanitize(html);
    } catch (e) {
        console.warn('Markdown render failed, falling back to plain text', e);
        return escapeHtml(text);
    }
}

function buildResultCard(r, allResults) {
    const filename = String(r.filename == null ? '' : r.filename);
    const score = Number(r.score) || 0;
    const card = document.createElement('div');
    card.className = 'result-card';
    card.dataset.filename = filename;
    card.addEventListener('click', () => viewDocument(card.dataset.filename, allResults));

    const header = document.createElement('div');
    header.className = 'result-header';
    const name = document.createElement('span');
    name.textContent = filename;
    const badge = document.createElement('span');
    badge.className = 'score-badge ' + (score > 0.8 ? 'match-high' : (score > 0.6 ? 'match-med' : ''));
    badge.textContent = (score * 100).toFixed(0) + '%';
    header.append(name, badge);

    const text = document.createElement('div');
    text.className = 'result-text';
    text.textContent = r.chunk_text == null ? '' : String(r.chunk_text);

    card.append(header, text);
    return card;
}

function addMessage(type, text, results = null, meta = null) {
    // Hide welcome hero on first interaction
    const hero = document.getElementById('welcome-hero');
    if (hero) hero.style.display = 'none';

    const history = document.getElementById('chat-history');
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${type}`;

    let contentHtml = `<div class="msg-bubble">${renderBubbleContent(type, text)}</div>`;

    if (results && results.length > 0) {
        contentHtml += `<div class="results-grid"></div>`;
    }

    if (type === 'bot' && meta && meta.total_ms != null) {
        const secs = (meta.total_ms / 1000).toFixed(1);
        const parts = [
            `<span class="meta-item"><svg fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>${secs}s</span>`
        ];
        if (meta.sources) parts.push(`<span class="meta-item">${meta.sources} source${meta.sources === 1 ? '' : 's'}</span>`);
        contentHtml += `<div class="msg-meta">${parts.join('<span>·</span>')}</div>`;
    }

    const label = type === 'user'
        ? `<div class="message-label">You</div>`
        : `<div class="message-label">Helios <span class="label-tag">Assistant</span></div>`;
    const avatar = type === 'user'
        ? `<div class="msg-avatar">You</div>`
        : `<div class="msg-avatar">${BOT_AVATAR_SVG}</div>`;

    msgDiv.innerHTML = `
        ${avatar}
        <div class="msg-main">
            ${label}
            ${contentHtml}
        </div>
    `;

    const grid = msgDiv.querySelector('.results-grid');
    if (grid && results) {
        results.forEach(r => grid.appendChild(buildResultCard(r, results)));
    }

    history.appendChild(msgDiv);
    history.scrollTop = history.scrollHeight;
}

function askSuggestion(text) {
    const input = document.getElementById('query-input');
    input.value = text;
    doSearch();
}

function escapeHtml(text) {
    if (!text) return '';
    return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function showToast(message, success = true, duration = 4000) {
    // Remove existing toast
    const existing = document.getElementById('toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.id = 'toast';
    toast.className = `toast ${success ? 'toast-success' : 'toast-error'}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    // Auto-dismiss
    setTimeout(() => {
        toast.classList.add('toast-hide');
        setTimeout(() => toast.remove(), 400);
    }, duration);
}

async function viewDocument(filename, results) {
    // The full-document endpoint is operator-only; the public view shows the
    // excerpts retrieved for this answer.
    const excerpts = (results || [])
        .filter(r => r.filename === filename)
        .map(r => r.chunk_text || '');
    const body = excerpts.length
        ? excerpts.join('\n\n---\n\n')
        : 'No excerpts available for this source.';
    openModal(filename, body + '\n\n— Showing retrieved excerpts only. The full document is available to operators.');
}

function openModal(title, text) {
    document.getElementById('modal-title').textContent = title;
    document.getElementById('modal-text').textContent = text;
    document.getElementById('doc-modal').classList.add('open');
}

function closeModal() {
    document.getElementById('doc-modal').classList.remove('open');
}

// Close modal on backdrop click
document.getElementById('doc-modal').addEventListener('click', (e) => {
    if (e.target.id === 'doc-modal') closeModal();
});

// Mobile sidebar toggle helpers
function toggleSidebar() {
    const sb = document.querySelector('.sidebar');
    const backdrop = document.querySelector('.sidebar-backdrop');
    if (!sb) return;
    const isOpen = sb.classList.contains('open');

    if (isOpen) {
        sb.classList.remove('open');
        backdrop && backdrop.classList.remove('show');
    } else {
        sb.classList.add('open');
        backdrop && backdrop.classList.add('show');
        // Prevent body scroll when sidebar is open on mobile
        if (window.innerWidth <= 768) {
            document.body.style.overflow = 'hidden';
        }
    }
}

function closeSidebar() {
    const sb = document.querySelector('.sidebar');
    const backdrop = document.querySelector('.sidebar-backdrop');
    if (!sb) return;
    sb.classList.remove('open');
    backdrop && backdrop.classList.remove('show');
    // Re-enable body scroll
    document.body.style.overflow = '';
}

// Close sidebar when clicking backdrop
document.querySelector('.sidebar-backdrop')?.addEventListener('click', closeSidebar);

// Ensure sidebar is closed when resizing to desktop
window.addEventListener('resize', () => {
    if (window.innerWidth > 768) {
        closeSidebar();
    }
});

// Close sidebar on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && window.innerWidth <= 768) {
        closeSidebar();
    }
});

// ---- Event wiring (replaces inline on*= attributes) ----
document.addEventListener('click', (e) => {
    const el = e.target.closest('[data-action], [data-view], [data-suggestion]');
    if (!el) return;
    if (el.dataset.action === 'toggle-sidebar') toggleSidebar();
    else if (el.dataset.action === 'new-conversation') newConversation();
    else if (el.dataset.action === 'attach-image') document.getElementById('chat-file-input').click();
    else if (el.dataset.action === 'close-modal') closeModal();
    else if (el.dataset.view) setView(el.dataset.view);
    else if (el.dataset.suggestion) askSuggestion(el.dataset.suggestion);
});
queryInput.addEventListener('keypress', handleKeyPress);
document.getElementById('chat-file-input').addEventListener('change', handleChatFileSelect);
document.getElementById('send-btn').addEventListener('click', doSearch);
