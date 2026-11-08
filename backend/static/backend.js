// Backend viewer — polling-based dashboard
// Public demo actions are read-only; private operator mode can perform guarded writes.

// ---------- top tabs ----------
function setTab(name) {
    document.querySelectorAll('.bk-tab').forEach(b => {
        b.classList.toggle('active', b.dataset.tab === name);
    });
    document.querySelectorAll('.bk-tab-panel').forEach(p => {
        p.hidden = p.id !== ('tab-' + name);
    });
    // Charts redraw correctly only when visible — nudge them when the usage tab opens
    if (name === 'usage') {
        Object.values(CHARTS).forEach(c => c && c.resize());
    }
    if (name === 'sources') {
        refreshSources();
    }
    if (name === 'conversations') {
        refreshConversations();
    }
}

function getAdminToken() {
    return localStorage.getItem('admin_token') || '';
}

function authHeaders() {
    const t = getAdminToken();
    return t ? { 'Authorization': 'Bearer ' + t } : {};
}

function isReadOnlyDemo() {
    return document.body.dataset.readOnly === 'true';
}

function escapeHtml(s) {
    return String(s == null ? '' : s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function truncate(s, n) {
    s = String(s == null ? '' : s);
    return s.length > n ? s.slice(0, n) + '…' : s;
}

// ---------- date/time formatting (single formatter) ----------

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const _pad2 = n => String(n).padStart(2, '0');

// "08 Nov 2026 14:32:05 CET" in the browser's local time. compact -> "08 Nov 14:32".
function fmtDateTime(iso, opts) {
    if (iso === null || iso === undefined || iso === '') return '—';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    const day = `${_pad2(d.getDate())} ${MONTHS[d.getMonth()]}`;
    if (opts && opts.compact) return `${day} ${_pad2(d.getHours())}:${_pad2(d.getMinutes())}`;
    let tz = '';
    try {
        const part = new Intl.DateTimeFormat('en-GB', { timeZoneName: 'short' })
            .formatToParts(d).find(x => x.type === 'timeZoneName');
        tz = part ? ' ' + part.value : '';
    } catch (_) { /* tz name is optional */ }
    return `${day} ${d.getFullYear()} ${_pad2(d.getHours())}:${_pad2(d.getMinutes())}:${_pad2(d.getSeconds())}${tz}`;
}

function fmtShort(iso) {
    return fmtDateTime(iso, { compact: true });
}

// ---------- small DOM helpers ----------

function h(tag, props, ...children) {
    const el = document.createElement(tag);
    if (props) {
        for (const [k, v] of Object.entries(props)) {
            if (v === null || v === undefined || v === false) continue;
            if (k === 'class') el.className = v;
            else if (k === 'text') el.textContent = v;
            else if (k === 'dataset') Object.assign(el.dataset, v);
            else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
            else el.setAttribute(k, v === true ? '' : v);
        }
    }
    children.flat().forEach(c => {
        if (c === null || c === undefined || c === false) return;
        el.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    });
    return el;
}

function messageRow(colspan, text, isError) {
    return h('tr', null, h('td', { colspan: String(colspan), class: 'conv-empty' + (isError ? ' conv-empty-error' : ''), text }));
}

async function copyText(text, label) {
    try {
        await navigator.clipboard.writeText(text);
        showToast(`${label} copied.`);
    } catch (_) {
        const ta = h('textarea', { style: 'position:fixed;opacity:0' });
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        let ok = false;
        try { ok = document.execCommand('copy'); } catch (__) { ok = false; }
        ta.remove();
        showToast(ok ? `${label} copied.` : 'Copy failed.', ok);
    }
}

// Promise-based confirm dialog (replaces window.confirm).
function confirmDialog(message, okLabel) {
    const modal = document.getElementById('confirm-modal');
    const ok = document.getElementById('confirm-modal-ok');
    const cancel = document.getElementById('confirm-modal-cancel');
    if (!modal || !ok || !cancel) return Promise.resolve(false);
    document.getElementById('confirm-modal-msg').textContent = message;
    ok.textContent = okLabel || 'Delete';
    modal.classList.add('open');
    cancel.focus();
    return new Promise(resolve => {
        const done = (val) => {
            modal.classList.remove('open');
            ok.removeEventListener('click', onOk);
            cancel.removeEventListener('click', onCancel);
            modal.removeEventListener('click', onBackdrop);
            document.removeEventListener('keydown', onKey, true);
            resolve(val);
        };
        const onOk = () => done(true);
        const onCancel = () => done(false);
        const onBackdrop = (e) => { if (e.target === modal) done(false); };
        const onKey = (e) => { if (e.key === 'Escape') { e.stopPropagation(); done(false); } };
        ok.addEventListener('click', onOk);
        cancel.addEventListener('click', onCancel);
        modal.addEventListener('click', onBackdrop);
        document.addEventListener('keydown', onKey, true);
    });
}

// ---------- conversations ----------

const CONV_PAGE_SIZE = 50;
const CONV = { items: [], total: 0, next: null, q: '', seq: 0, loading: false, error: null };

function canDelete() {
    return !isReadOnlyDemo() && Boolean(getAdminToken());
}

function updateConvChrome() {
    const note = document.getElementById('conv-auth-note');
    if (note) {
        note.textContent = isReadOnlyDemo()
            ? 'Read-only demo: deletion is disabled.'
            : (getAdminToken() ? 'Operator token saved — delete actions enabled.' : 'No operator token saved.');
    }
    const clear = document.getElementById('conv-clear-btn');
    if (clear) clear.hidden = !canDelete();
    const more = document.getElementById('conv-more-btn');
    if (more) {
        more.hidden = CONV.next === null || CONV.error !== null;
        more.disabled = CONV.loading;
    }
    const count = document.getElementById('conv-count');
    if (count) count.textContent = CONV.error ? '' : `Showing ${CONV.items.length} of ${CONV.total}`;
}

function renderConversations() {
    const tbody = document.getElementById('conv-tbody');
    if (!tbody) return;
    tbody.replaceChildren();
    if (CONV.error === 'unauthorized') {
        const msg = isReadOnlyDemo()
            ? 'Operator access required — conversations are not available in the read-only demo.'
            : (getAdminToken()
                ? 'Operator token rejected — re-save it under Sources.'
                : 'Operator token required — save it under Sources');
        tbody.appendChild(messageRow(9, msg, false));
    } else if (CONV.error === 'unavailable') {
        tbody.appendChild(messageRow(9, 'Conversation store unavailable', true));
    } else if (CONV.error) {
        tbody.appendChild(messageRow(9, 'Failed to load conversations: ' + CONV.error, true));
    } else if (CONV.items.length === 0) {
        tbody.appendChild(messageRow(9, CONV.loading ? 'Loading…'
            : (CONV.q ? 'No conversations match your search.' : 'No conversations yet.'), false));
    } else {
        const showDelete = canDelete();
        CONV.items.forEach(r => tbody.appendChild(convRow(r, showDelete)));
    }
    updateConvChrome();
}

function convRow(r, showDelete) {
    const tr = h('tr', { class: 'conv-row', tabindex: '0', role: 'button', 'aria-label': 'Open conversation details' });
    const tIn = r.tokens_input | 0, tOut = r.tokens_output | 0;
    tr.append(
        h('td', { class: 'conv-date', title: r.ts || '', text: fmtDateTime(r.ts) }),
        h('td', { class: 'mono', title: r.thread_id || '', text: (r.thread_id || '').slice(0, 8) || '—' }),
        h('td', { title: truncate(r.query, 400), text: truncate(r.query, 120) }),
        h('td', { title: truncate(r.answer, 400), text: truncate(r.answer, 160) }),
        h('td', { class: 'num', text: `${r.total_ms | 0} ms` }),
        h('td', { class: 'num', text: `${tIn.toLocaleString('en-GB')} / ${tOut.toLocaleString('en-GB')}` }),
        h('td', { class: 'num', text: String(r.doc_count | 0) }),
        h('td', { text: r.model || '—' }),
    );
    const actions = h('td');
    if (showDelete) {
        actions.appendChild(h('button', {
            class: 'conv-del', title: 'Delete conversation', 'aria-label': 'Delete conversation',
            text: '×',
            onclick: (e) => { e.stopPropagation(); deleteConversation(r.id); },
        }));
    } else if (isReadOnlyDemo()) {
        actions.appendChild(h('span', { class: 'conv-readonly', text: 'locked' }));
    }
    tr.appendChild(actions);
    tr.addEventListener('click', () => openConvModal(r));
    tr.addEventListener('keydown', (e) => {
        if (e.target === tr && (e.key === 'Enter' || e.key === ' ')) {
            e.preventDefault();
            openConvModal(r);
        }
    });
    return tr;
}

async function fetchConversationsPage(beforeId) {
    const params = new URLSearchParams({ limit: String(CONV_PAGE_SIZE) });
    if (beforeId !== null && beforeId !== undefined) params.set('before_id', String(beforeId));
    if (CONV.q) params.set('q', CONV.q);
    const r = await fetch('/backend/api/conversations?' + params.toString(), {
        credentials: 'same-origin',
        headers: authHeaders(),
    });
    if (r.status === 401) return { error: 'unauthorized' };
    if (r.status === 500) return { error: 'unavailable' };
    if (!r.ok) return { error: 'HTTP ' + r.status };
    return { data: await r.json() };
}

async function loadConversations(append) {
    const seq = ++CONV.seq;
    CONV.loading = true;
    if (!append) {
        CONV.items = [];
        CONV.next = null;
        CONV.error = null;
    }
    renderConversations();
    let res;
    try {
        res = await fetchConversationsPage(append ? CONV.next : null);
    } catch (e) {
        res = { error: e.message || 'network error' };
    }
    if (seq !== CONV.seq) return; // a newer request superseded this one
    CONV.loading = false;
    if (res.error) {
        CONV.error = res.error;
        CONV.items = [];
        CONV.next = null;
        CONV.total = 0;
    } else {
        CONV.error = null;
        const data = res.data;
        setText('conv-backend', data.backend || '—');
        CONV.items = append ? CONV.items.concat(data.items || []) : (data.items || []);
        CONV.total = data.total | 0;
        CONV.next = data.next_before_id === undefined ? null : data.next_before_id;
    }
    renderConversations();
}

function refreshConversations() {
    const input = document.getElementById('conv-search');
    CONV.q = input ? input.value.trim() : '';
    return loadConversations(false);
}

function loadMoreConversations() {
    if (CONV.next === null || CONV.loading) return;
    return loadConversations(true);
}

let _convSearchTimer = null;
function initConversationControls() {
    const input = document.getElementById('conv-search');
    if (input) {
        input.addEventListener('input', () => {
            clearTimeout(_convSearchTimer);
            _convSearchTimer = setTimeout(refreshConversations, 300);
        });
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                clearTimeout(_convSearchTimer);
                refreshConversations();
            }
        });
    }
    const modal = document.getElementById('conv-modal');
    if (modal) modal.addEventListener('click', (e) => { if (e.target === modal) closeConvModal(); });
    document.addEventListener('keydown', (e) => {
        if (e.key !== 'Escape') return;
        if (modal && modal.classList.contains('open')) closeConvModal();
        else closeSourceModal();
    });
}

function openConvModal(r) {
    const body = document.getElementById('conv-modal-body');
    const modal = document.getElementById('conv-modal');
    if (!body || !modal) return;
    body.replaceChildren();
    document.getElementById('conv-modal-title').textContent = `Conversation #${r.id ?? ''}`;

    const meta = (label, value) => h('div', null, h('dt', { text: label }), h('dd', null, value));
    const threadValue = r.thread_id
        ? h('button', {
            class: 'conv-thread-link', title: 'Filter conversations by this thread', text: r.thread_id,
            onclick: () => {
                const input = document.getElementById('conv-search');
                if (input) input.value = r.thread_id;
                closeConvModal();
                refreshConversations();
            },
        })
        : '—';
    body.appendChild(h('dl', { class: 'conv-detail-meta' },
        meta('Time', fmtDateTime(r.ts)),
        meta('ISO', r.ts || '—'),
        meta('Thread', threadValue),
        meta('Model', r.model || '—'),
        meta('Embed', `${r.embed_ms | 0} ms`),
        meta('AI', `${r.ai_ms | 0} ms`),
        meta('Total', `${r.total_ms | 0} ms`),
        meta('Tokens in / out', `${(r.tokens_input | 0).toLocaleString('en-GB')} / ${(r.tokens_output | 0).toLocaleString('en-GB')}`),
        meta('Docs', String(r.doc_count | 0)),
    ));
    const block = (title, text, label) => h('section', { class: 'conv-detail-block' },
        h('h4', null, title, h('button', {
            class: 'conv-btn conv-btn-neutral conv-btn-sm', text: 'Copy',
            onclick: () => copyText(text, label),
        })),
        h('pre', { class: 'conv-detail-text', text: text || '—' }));
    body.appendChild(block('Question', r.query || '', 'Question'));
    body.appendChild(block('Answer', r.answer || '', 'Answer'));
    modal.classList.add('open');
    modal.querySelector('.close-btn')?.focus();
}

function closeConvModal() {
    document.getElementById('conv-modal')?.classList.remove('open');
}

async function deleteConversation(id) {
    if (isReadOnlyDemo()) {
        showToast('Read-only demo: conversation deletion is disabled.', false);
        return;
    }
    if (!getAdminToken()) {
        showToast('Operator token required — save it under Sources.', false);
        return;
    }
    if (!await confirmDialog(`Delete conversation #${id}?`)) return;
    try {
        const r = await fetch('/backend/api/conversations/' + encodeURIComponent(id), {
            method: 'DELETE',
            headers: authHeaders(),
            credentials: 'same-origin',
        });
        if (r.status === 401) { showToast('Unauthorized — check the operator token.', false); return; }
        if (!r.ok) throw new Error('HTTP ' + r.status);
        showToast('Conversation deleted.');
        refreshConversations();
    } catch (e) {
        showToast('Delete failed: ' + e.message, false);
    }
}

async function clearAllConversations() {
    if (isReadOnlyDemo()) {
        showToast('Read-only demo: conversation deletion is disabled.', false);
        return;
    }
    if (!getAdminToken()) {
        showToast('Operator token required — save it under Sources.', false);
        return;
    }
    if (!await confirmDialog('Delete ALL conversations? This cannot be undone.', 'Delete all')) return;
    try {
        const r = await fetch('/backend/api/conversations', {
            method: 'DELETE',
            headers: authHeaders(),
            credentials: 'same-origin',
        });
        if (r.status === 401) { showToast('Unauthorized — check the operator token.', false); return; }
        if (!r.ok) throw new Error('HTTP ' + r.status);
        const data = await r.json().catch(() => ({}));
        showToast(`Deleted ${data.deleted ?? 0} conversation(s).`);
        refreshConversations();
    } catch (e) {
        showToast('Clear failed: ' + e.message, false);
    }
}
const SYS_POLL_MS = 30_000;
const METRICS_POLL_MS = 5_000;
const CONN_POLL_MS = 5_000;
const USAGE_POLL_MS = 30_000;

let SYSTEM_STATE = null;
let PROMPT_STATE = null;
let MODELS_STATE = null;
let SOURCE_FILES = [];
const CHARTS = {};

// ---------- shared helpers ----------
function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value === undefined || value === null ? '—' : String(value);
}

function clampPct(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return 0;
    return Math.max(0, Math.min(100, n));
}

function setWidth(id, pct) {
    const el = document.getElementById(id);
    if (el) el.style.width = `${clampPct(pct)}%`;
}

function setSparkHeight(id, pct) {
    const el = document.getElementById(id);
    if (el) el.style.height = `${Math.max(10, clampPct(pct))}%`;
}

function setRing(id, pct) {
    const el = document.getElementById(id);
    const ring = el?.parentElement;
    if (ring) ring.style.setProperty('--budget-deg', `${clampPct(pct) * 3.6}deg`);
}

function showToast(message, success = true, duration = 4000) {
    const existing = document.getElementById('toast');
    if (existing) existing.remove();
    const toast = document.createElement('div');
    toast.id = 'toast';
    toast.className = `toast ${success ? 'toast-success' : 'toast-error'}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.classList.add('toast-hide');
        setTimeout(() => toast.remove(), 400);
    }, duration);
}

function setDot(id, color) {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.remove('tile-dot-green', 'tile-dot-yellow', 'tile-dot-amber', 'tile-dot-red');
    if (color) el.classList.add(`tile-dot-${color === 'yellow' ? 'amber' : color}`);
}

async function jget(path) {
    const r = await fetch(path, { credentials: 'same-origin' });
    if (!r.ok) throw new Error(`${path}: ${r.status}`);
    return r.json();
}

function humanSeconds(s) {
    if (!s || s < 60) return `${s|0}s`;
    if (s < 3600) return `${(s/60)|0}m`;
    if (s < 86400) return `${(s/3600)|0}h ${((s%3600)/60)|0}m`;
    return `${(s/86400)|0}d`;
}

// ---------- pollers ----------

async function pollSystem() {
    const s = await jget('/backend/api/system');
    SYSTEM_STATE = s;
    setText('tile-primary-model', s.model_primary);
    setText('tile-fallback-model', s.model_fallback);
    setText('tile-embed-model', s.embedding_model);
    setText('tile-embed-dim', `${s.embedding_provider} · ${s.embedding_dim} dim`);
    setText('tile-region', s.llm_provider);
    setText('tile-git-sha', s.git_sha);
    setText('tile-uptime', `uptime ${humanSeconds(s.uptime_seconds)}`);
    setText('runtime-provider', s.llm_provider);
    setText('runtime-embed', truncate(s.embedding_model, 28));
    setText('runtime-primary', truncate(s.model_primary, 28));
    document.getElementById('config-pre').textContent = JSON.stringify(s.config, null, 2);
}

async function pollConnections() {
    const c = await jget('/backend/api/connections');
    const okStatuses = {
        qdrant: ['up'],
        gemini: ['up'],
        embedding_model: ['loaded', 'configured'],
        logger: ['writing', 'stdout_only'],
        usage_db: ['up', 'memory_fallback'],
        chat_db: ['up', 'memory_fallback'],
    };
    const labels = {
        chat_db: 'Chat DB',
        usage_db: 'Usage DB',
        embedding_model: 'Embedding',
    };
    Object.entries(c).forEach(([key, val]) => {
        const span = document.querySelector(`[data-conn="${key}"]`);
        if (!span) return;
        const accepted = okStatuses[key] || ['up'];
        const good = accepted.includes(val.status);
        const dot = good ? '🟢' : '🔴';
        const label = labels[key] || key.replace(/_/g, ' ');
        const suffix = (key === 'chat_db' && val.count != null) ? ` (${val.count})` : '';
        span.textContent = `${dot} ${label}${suffix}`;
    });
    // Tile dots
    setDot('tile-qdrant-dot', c.qdrant?.status === 'up' ? 'green' : 'red');
    const qdrantPoints = c.qdrant?.points_count;
    const qdrantLabel = typeof qdrantPoints === 'number' ? qdrantPoints.toLocaleString('en-GB') : (qdrantPoints ?? '—');
    setText('tile-qdrant-points', qdrantLabel);
    setText('runtime-qdrant-points', qdrantLabel);
    setDot('tile-embed-dot', ['loaded', 'configured'].includes(c.embedding_model?.status) ? 'green' : 'yellow');
    const geminiColor = c.gemini?.status === 'up' ? 'green' : (c.gemini?.status === 'unconfigured' ? 'yellow' : 'red');
    setDot('tile-primary-dot', geminiColor);
    setDot('tile-fallback-dot', geminiColor);
}

// ---------- RAG source library ----------

function initSourceControls() {
    const tokenInput = document.getElementById('operator-token-input');
    if (tokenInput) tokenInput.value = getAdminToken();

    const dropZone = document.getElementById('source-drop-zone');
    if (!dropZone) return;
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files) {
            addSourcesToQueue(Array.from(e.dataTransfer.files));
        }
    });
}

function saveOperatorToken() {
    const input = document.getElementById('operator-token-input');
    if (!input) {
        showToast('Operator token is hidden in the read-only demo.', false);
        return;
    }
    localStorage.setItem('admin_token', input.value.trim());
    showToast('Operator token saved.');
    refreshSources();
    refreshConversations();
}

function handleSourceFileSelect(e) {
    addSourcesToQueue(Array.from(e.target.files || []));
    e.target.value = '';
}

function addSourcesToQueue(files) {
    SOURCE_FILES = files;
    const list = document.getElementById('source-file-list');
    const queue = document.getElementById('source-upload-queue');
    if (!list || !queue) return;
    list.innerHTML = files.map(f => `<div>${escapeHtml(f.name)} (${(f.size / 1024).toFixed(1)} KB)</div>`).join('');
    queue.hidden = files.length === 0;
}

async function uploadSources() {
    if (isReadOnlyDemo()) {
        showToast('Upload is disabled in the read-only demo.', false);
        return;
    }
    if (SOURCE_FILES.length === 0) return;
    const token = getAdminToken();
    if (!token) {
        showToast('Private operator token required to upload sources.', false);
        return;
    }
    const btn = document.getElementById('source-upload-btn');
    const progress = document.getElementById('source-upload-progress');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Uploading...';
    }
    const errors = [];
    for (let i = 0; i < SOURCE_FILES.length; i++) {
        const file = SOURCE_FILES[i];
        if (progress) progress.textContent = `Uploading ${i + 1}/${SOURCE_FILES.length}: ${file.name}`;
        const fd = new FormData();
        fd.append('file', file);
        try {
            const res = await fetch('/upload', { method: 'POST', body: fd, headers: authHeaders() });
            if (res.status === 401) throw new Error('Unauthorized');
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
        } catch (e) {
            errors.push(`${file.name}: ${e.message}`);
        }
    }
    if (btn) {
        btn.disabled = false;
        btn.textContent = 'Start upload';
    }
    SOURCE_FILES = [];
    const queue = document.getElementById('source-upload-queue');
    if (queue) queue.hidden = true;
    const input = document.getElementById('source-file-input');
    if (input) input.value = '';
    if (errors.length) showToast(`Upload completed with ${errors.length} error(s).`, false);
    else showToast('Sources uploaded.');
    refreshSources();
}

async function clearSources() {
    if (isReadOnlyDemo()) {
        showToast('Clear is disabled in the read-only demo.', false);
        return;
    }
    if (!getAdminToken()) {
        showToast('Private operator token required to clear sources.', false);
        return;
    }
    if (!await confirmDialog('Delete ALL indexed sources? This cannot be undone.', 'Delete all')) return;
    try {
        const res = await fetch('/reset', { method: 'POST', headers: authHeaders() });
        if (res.status === 401) {
            showToast('Unauthorized. Check the operator token.', false);
            return;
        }
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
        showToast(data.deleted !== undefined ? `Cleared ${data.deleted} points.` : 'Sources cleared.');
        refreshSources();
    } catch (e) {
        showToast('Clear failed: ' + e.message, false);
    }
}

async function optimizeSources(e, btnEl) {
    if (isReadOnlyDemo()) {
        showToast('Optimize is disabled in the read-only demo.', false);
        return;
    }
    const btn = btnEl || (e && e.target && e.target.closest && e.target.closest('button'));
    const original = btn ? btn.innerHTML : '';
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Optimizing...';
    }
    try {
        const res = await fetch('/optimize', { method: 'POST', headers: authHeaders() });
        if (res.status === 401) {
            showToast('Unauthorized. Check the operator token.', false);
            return;
        }
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
        showToast(data.message || 'Optimization triggered.');
    } catch (e) {
        showToast('Optimize failed: ' + e.message, false);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = original;
        }
    }
}

function sourcesMessage(text, isError) {
    return h('tr', null, h('td', { colspan: '5', class: 'sources-note', style: isError ? 'color:var(--danger);' : null, text }));
}

async function refreshSources() {
    const tbody = document.getElementById('sources-list');
    if (!tbody) return;
    const badge = document.getElementById('source-status-badge');
    try {
        const res = await fetch('/documents', { credentials: 'same-origin', headers: authHeaders() });
        if (res.status === 401) {
            tbody.replaceChildren(sourcesMessage(isReadOnlyDemo()
                ? 'Operator access required — the source list is not available in the read-only demo.'
                : (getAdminToken()
                    ? 'Operator token rejected — re-save it in the Private operator token field.'
                    : 'Operator token required — save it in the Private operator token field.')));
            if (badge) { badge.textContent = 'token required'; badge.className = 'badge badge-txt'; }
            return;
        }
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const docs = await res.json();
        if (badge) {
            badge.textContent = `${docs.length} source${docs.length === 1 ? '' : 's'}`;
            badge.className = 'badge badge-csv';
        }
        if (docs.length === 0) {
            tbody.replaceChildren(sourcesMessage('No indexed sources are visible yet.'));
            return;
        }
        tbody.replaceChildren(...docs.map(sourceRow));
    } catch (e) {
        tbody.replaceChildren(sourcesMessage('Failed to load sources: ' + e.message, true));
    }
}

function sourceRow(d) {
    let badgeClass = 'badge-txt';
    if (d.type === 'PDF') badgeClass = 'badge-pdf';
    if (d.type === 'DOCX') badgeClass = 'badge-docx';
    if (d.type === 'CSV') badgeClass = 'badge-csv';
    const view = h('button', {
        class: 'btn btn-secondary btn-sm', title: 'View source', text: 'View',
        onclick: () => viewSourceDocument(d.id, d.filename),
    });
    const actions = h('td', { class: 'actions-cell' });
    if (isReadOnlyDemo()) {
        actions.append(view, h('span', { class: 'locked-action', text: 'locked' }));
    } else {
        actions.append(h('button', {
            class: 'btn btn-danger btn-sm', title: 'Delete source', text: 'Delete',
            onclick: () => deleteSource(d.id),
        }), view);
    }
    return h('tr', null,
        h('td', { 'data-label': 'File' }, h('strong', { text: d.filename || d.id || '—' })),
        h('td', { 'data-label': 'Type' }, h('span', { class: 'badge ' + badgeClass, text: d.type || '—' })),
        h('td', { 'data-label': 'Indexed', text: d.uploaded_at ? fmtDateTime(d.uploaded_at) : 'N/A' }),
        h('td', { 'data-label': 'Chunks', text: d.chunks === undefined || d.chunks === null ? '—' : String(d.chunks) }),
        actions);
}

async function deleteSource(id) {
    if (isReadOnlyDemo()) {
        showToast('Delete is disabled in the read-only demo.', false);
        return;
    }
    if (!getAdminToken()) {
        showToast('Private operator token required to delete sources.', false);
        return;
    }
    if (!await confirmDialog('Delete this indexed source?')) return;
    try {
        const res = await fetch('/document/' + encodeURIComponent(id), { method: 'DELETE', headers: authHeaders() });
        if (res.status === 401) {
            showToast('Unauthorized. Check the operator token.', false);
            return;
        }
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        showToast('Source deleted.');
        refreshSources();
    } catch (e) {
        showToast('Delete failed: ' + e.message, false);
    }
}

async function viewSourceDocument(id, filename) {
    openSourceModal('Loading...', 'Fetching source...');
    try {
        let res = await fetch('/document/' + encodeURIComponent(id || filename), { headers: authHeaders() });
        if (res.status === 401) {
            openSourceModal('Operator token required', 'Save the operator token to view source content.');
            return;
        }
        if (!res.ok && filename && id !== filename) {
            res = await fetch('/document/' + encodeURIComponent(filename), { headers: authHeaders() });
        }
        if (!res.ok) throw new Error('Source not found');
        const data = await res.json();
        openSourceModal(data.filename || filename || id, data.text || '');
    } catch (e) {
        openSourceModal('Error', 'Could not load source content. ' + e.message);
    }
}

function openSourceModal(title, text) {
    document.getElementById('source-modal-title').textContent = title;
    document.getElementById('source-modal-text').textContent = text;
    document.getElementById('source-modal').classList.add('open');
}

function closeSourceModal() {
    document.getElementById('source-modal').classList.remove('open');
}

document.getElementById('source-modal')?.addEventListener('click', (e) => {
    if (e.target.id === 'source-modal') closeSourceModal();
});

async function pollMetrics() {
    const m = await jget('/backend/api/metrics');
    const memoryMb = Number(m.memory_mb || 0);
    const latencyMs = Number(m.avg_latency_ms || 0);
    const cachePct = Number(m.cache_hit_ratio_pct || 0);
    const requestsTotal = Number(m.requests_total || 0);

    setText('m-memory', memoryMb.toFixed(0));
    setText('m-latency', latencyMs.toFixed(0));
    setText('m-cache', cachePct.toFixed(0));
    setText('m-requests', requestsTotal.toLocaleString('en-GB'));
    setText('tile-cache-ratio', `${m.cache_hit_ratio_pct}%`);
    setText('tile-fallback-count', m.fallback_invocations);
    setWidth('g-memory-bar', (memoryMb / 1024) * 100);
    setWidth('g-latency-bar', (latencyMs / 3000) * 100);
    setWidth('g-cache-bar', cachePct);
    setSparkHeight('g-requests-bar', (Math.log10(requestsTotal + 1) / 3) * 100);
    setText('g-memory-meta', `${memoryMb.toFixed(0)} MB of a 1 GB soft ceiling`);
    setText('g-latency-meta', latencyMs > 3000 ? 'Above 3,000 ms target' : 'Target under 3,000 ms');
    setText('g-cache-meta', `${cachePct.toFixed(0)}% prompt/model cache reuse`);
    setText('g-requests-meta', `${requestsTotal.toLocaleString('en-GB')} handled requests`);

    const rl = m.rate_limit || {};
    const used = rl.tokens_used_today || 0;
    const budget = rl.daily_token_budget || 0;
    setText('tile-tokens-used', `${used.toLocaleString('en-GB')} / ${budget.toLocaleString('en-GB')}`);
    setText('tile-budget-reset', fmtShort(rl.budget_resets_at));
    setText('g-token-readout', `${used.toLocaleString('en-GB')} / ${budget.toLocaleString('en-GB')}`);
    const budgetPct = budget > 0 ? (used / budget) * 100 : 0;
    setWidth('g-token-bar', budgetPct);
    setRing('g-token-ring', budgetPct);
    setText('g-token-meta', `${clampPct(budgetPct).toFixed(0)}% used. Resets ${fmtShort(rl.budget_resets_at)}`);
    let budgetColor = 'green';
    if (budget > 0) {
        const ratio = used / budget;
        if (ratio > 0.95) budgetColor = 'red';
        else if (ratio > 0.7) budgetColor = 'yellow';
    }
    setDot('tile-budget-dot', budgetColor);
}

async function pollModels() {
    const data = await jget('/backend/api/models');
    MODELS_STATE = data;
    const primary = data.models.find(m => m.role === 'primary') || data.models[0];
    const fallback = data.models.find(m => m.role === 'fallback');
    setText('dd-primary-active', primary ? `${primary.label} — $${primary.input_per_m_usd}/$${primary.output_per_m_usd} per M` : 'unconfigured');
    setText('dd-fallback-active', fallback ? `${fallback.label} — $${fallback.input_per_m_usd}/$${fallback.output_per_m_usd} per M` : 'unconfigured');
    renderModelList('dd-primary-list', data.models, 'primary');
    renderModelList('dd-fallback-list', data.models, 'fallback');
}

function renderModelList(id, models, activeRole) {
    const el = document.getElementById(id);
    if (!el) return;
    el.replaceChildren(...models.map(m => {
        const active = m.role === activeRole;
        const ctx = Number(m.context_window);
        return h('div', { class: 'model-row' + (active ? ' active' : '') },
            h('div', { class: 'model-row-name' },
                m.label || '—',
                active ? h('span', { class: 'model-role', text: 'ACTIVE' }) : null,
                h('span', { style: 'font-size:0.7rem;color:#94a3b8;', text: `(${m.vendor || '—'})` })),
            h('div', {
                class: 'model-row-meta',
                text: `$${m.input_per_m_usd}/M input · $${m.output_per_m_usd}/M output` +
                    (Number.isFinite(ctx) ? ` · ${ctx.toLocaleString()} ctx` : ''),
            }),
            h('div', { class: 'model-row-desc', text: m.description || '' }));
    }));
}

function toggleDropdown(which) {
    const el = document.getElementById(`dd-${which}-list`);
    if (el) el.hidden = !el.hidden;
}

// ---------- prompt playground ----------

async function loadPrompt() {
    try {
        const p = await jget('/backend/api/prompt');
        PROMPT_STATE = p;
        const ta = document.getElementById('prompt-system');
        if (ta && !ta.value) ta.value = p.system_prompt;
        updatePreview();
    } catch (e) {
        console.warn('loadPrompt failed', e);
    }
}

function updatePreview() {
    if (!PROMPT_STATE) return;
    const sys = document.getElementById('prompt-system').value;
    const example = PROMPT_STATE.example_context;
    const query = PROMPT_STATE.example_query;
    const userMsg = `Konteksten:\n${example}\n\nSpørgsmål: ${query}`;
    const preview = `system:\n${sys.slice(0, 500)}…\n\nuser:\n${userMsg}\n\n(JSON schema: ${PROMPT_STATE.schema.name}, strict=${PROMPT_STATE.schema.strict})`;
    document.getElementById('prompt-preview').textContent = preview;
}

function importPrompt(ev) {
    const file = ev.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
        document.getElementById('prompt-system').value = reader.result;
        updatePreview();
    };
    reader.readAsText(file);
}

function exportPrompt() {
    const text = document.getElementById('prompt-system').value;
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'system-prompt.txt';
    a.click();
    URL.revokeObjectURL(url);
}

// ---------- usage history charts ----------

function ensureChart(id, label, color) {
    if (CHARTS[id]) return CHARTS[id];
    const ctx = document.getElementById(id);
    if (!ctx) return null;
    CHARTS[id] = new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: [{
            label, data: [], borderColor: color, backgroundColor: color + '33',
            tension: 0.25, pointRadius: 0,
        }]},
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: '#94a3b8', maxRotation: 0 } },
                y: { ticks: { color: '#94a3b8' }, beginAtZero: true },
            },
        },
    });
    return CHARTS[id];
}

async function pollUsage() {
    const summary = await jget('/backend/api/usage/summary');
    document.getElementById('usage-backend').textContent = summary.backend || '—';
    const sumText = `Last 24h: ${summary.req_24h ?? 0} requests, ${(summary.tok_in_24h||0).toLocaleString()} input + ${(summary.tok_out_24h||0).toLocaleString()} output tokens, ${summary.errors_24h ?? 0} errors. Last 7d: ${summary.req_7d ?? 0} requests.`;
    document.getElementById('usage-summary').textContent = sumText;

    const [req, tok, cache, fb] = await Promise.all([
        jget('/backend/api/usage/timeseries?metric=requests&range=24h'),
        jget('/backend/api/usage/timeseries?metric=tokens&range=24h'),
        jget('/backend/api/usage/timeseries?metric=cache_hit_ratio&range=7d'),
        jget('/backend/api/usage/timeseries?metric=fallback_rate&range=7d'),
    ]);

    applyToChart('chart-requests', 'Requests', '#3b82f6', req.series);
    applyToChart('chart-tokens', 'Tokens', '#10b981', tok.series);
    applyToChart('chart-cache', 'Cache-hit %', '#f59e0b', cache.series);
    applyToChart('chart-fallback', 'Fallback %', '#ef4444', fb.series);
}

function applyToChart(id, label, color, series) {
    const c = ensureChart(id, label, color);
    if (!c) return;
    const wrap = c.canvas.parentElement;
    let empty = wrap.querySelector('.chart-empty');
    if (!series || series.length === 0) {
        if (!empty) wrap.appendChild(h('div', { class: 'chart-empty', text: 'No data for this period yet.' }));
        series = [];
    } else if (empty) {
        empty.remove();
    }
    c.data.labels = series.map(p => fmtShort(p.ts));
    c.data.datasets[0].data = series.map(p => p.value);
    c.update('none');
}

// ---------- poll scheduler ----------
// One tick loop for every poller: skips while the tab is hidden, never overlaps a
// poller with itself, and surfaces failures as a header "stale since" indicator.

const POLLERS = [
    { name: 'system',      fn: pollSystem,      every: SYS_POLL_MS },
    { name: 'connections', fn: pollConnections, every: CONN_POLL_MS },
    { name: 'metrics',     fn: pollMetrics,     every: METRICS_POLL_MS },
    { name: 'usage',       fn: pollUsage,       every: USAGE_POLL_MS },
    { name: 'models',      fn: pollModels,      every: Infinity }, // once; retried only after a failure
].map(p => ({ ...p, running: false, last: 0, failedSince: null }));

const POLL_RETRY_MS = 30_000;

function updateStaleIndicator() {
    const el = document.getElementById('stale-indicator');
    if (!el) return;
    const failed = POLLERS.filter(p => p.failedSince).map(p => p.failedSince);
    if (failed.length === 0) { el.hidden = true; return; }
    const since = new Date(Math.min(...failed.map(d => d.getTime())));
    const names = POLLERS.filter(p => p.failedSince).map(p => p.name).join(', ');
    el.textContent = `⚠ stale since ${_pad2(since.getHours())}:${_pad2(since.getMinutes())}`;
    el.title = `Failing: ${names}`;
    el.hidden = false;
}

async function runPoller(p) {
    if (p.running) return;
    p.running = true;
    p.last = Date.now();
    try {
        await p.fn();
        p.failedSince = null;
    } catch (e) {
        console.warn(`poll ${p.name} failed`, e);
        if (!p.failedSince) p.failedSince = new Date();
    } finally {
        p.running = false;
        updateStaleIndicator();
    }
}

function pollTick() {
    if (document.hidden) return;
    const now = Date.now();
    POLLERS.forEach(p => {
        const interval = p.failedSince ? Math.min(p.every, POLL_RETRY_MS) : p.every;
        if (p.last === 0 || now - p.last >= interval) runPoller(p);
    });
}

// ---------- bootstrap ----------

// Static markup carries data-* attributes instead of inline on*= handlers (CSP: no inline JS).
const ACTIONS = {
    'optimize-sources': (e, el) => optimizeSources(e, el),
    'clear-sources': () => clearSources(),
    'refresh-sources': () => refreshSources(),
    'pick-source-file': () => document.getElementById('source-file-input').click(),
    'upload-sources': () => uploadSources(),
    'save-operator-token': () => saveOperatorToken(),
    'toggle-dropdown': (e, el) => toggleDropdown(el.dataset.dropdown),
    'pick-prompt-file': () => document.getElementById('prompt-import').click(),
    'export-prompt': () => exportPrompt(),
    'update-preview': () => updatePreview(),
    'refresh-conversations': () => refreshConversations(),
    'clear-conversations': () => clearAllConversations(),
    'load-more-conversations': () => loadMoreConversations(),
    'close-source-modal': () => closeSourceModal(),
    'close-conv-modal': () => closeConvModal(),
};

function initActionHandlers() {
    document.addEventListener('click', (e) => {
        const tab = e.target.closest('.bk-tab[data-tab]');
        if (tab) { setTab(tab.dataset.tab); return; }
        const el = e.target.closest('[data-action]');
        const fn = el && ACTIONS[el.dataset.action];
        if (fn) fn(e, el);
    });
    document.getElementById('source-file-input')?.addEventListener('change', handleSourceFileSelect);
    document.getElementById('prompt-import')?.addEventListener('change', importPrompt);
}

async function bootstrap() {
    initActionHandlers();
    initSourceControls();
    initConversationControls();
    updateConvChrome();
    // Prompt playground depends on nothing polled; run it alongside the first tick.
    loadPrompt();
    refreshSources();
    pollTick();
    setInterval(pollTick, 1000);
    document.addEventListener('visibilitychange', () => { if (!document.hidden) pollTick(); });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bootstrap);
} else {
    bootstrap();
}
