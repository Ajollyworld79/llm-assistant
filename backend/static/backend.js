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
    return Boolean(window.HELIOS_READ_ONLY);
}

function escapeHtml(s) {
    return String(s == null ? '' : s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function escapeJsString(s) {
    return String(s == null ? '' : s)
        .replace(/\\/g, '\\\\')
        .replace(/'/g, "\\'")
        .replace(/\n/g, '\\n')
        .replace(/\r/g, '\\r');
}

function truncate(s, n) {
    s = String(s == null ? '' : s);
    return s.length > n ? s.slice(0, n) + '…' : s;
}

function shortTs(iso) {
    if (!iso) return '—';
    try {
        const d = new Date(iso);
        const pad = n => String(n).padStart(2, '0');
        return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
    } catch (e) { return iso; }
}

async function refreshConversations() {
    const tbody = document.getElementById('conv-tbody');
    const note = document.getElementById('conv-auth-note');
    if (note) note.textContent = isReadOnlyDemo()
        ? 'Read-only demo: history is inspectable, deletion is disabled.'
        : (getAdminToken()
            ? 'Admin token detected — delete actions enabled.'
            : 'No admin token saved. Delete actions remain disabled.');
    if (!tbody) return;
    try {
        const data = await jget('/backend/api/conversations?limit=200');
        setText('conv-backend', data.backend || '—');
        const items = data.items || [];
        if (items.length === 0) {
            tbody.innerHTML = '<tr><td colspan="10" style="padding:1rem; text-align:center; color:#888">No conversations yet.</td></tr>';
            return;
        }
        tbody.innerHTML = items.map(r => `
            <tr>
                <td title="${escapeHtml(r.ts || '')}">${escapeHtml(shortTs(r.ts))}</td>
                <td class="mono" title="${escapeHtml(r.thread_id || '')}">${escapeHtml((r.thread_id || '').slice(0, 8))}</td>
                <td title="${escapeHtml(r.query || '')}">${escapeHtml(truncate(r.query, 120))}</td>
                <td title="${escapeHtml(r.answer || '')}">${escapeHtml(truncate(r.answer, 160))}</td>
                <td class="num">${r.embed_ms|0} ms</td>
                <td class="num">${r.ai_ms|0} ms</td>
                <td class="num">${r.total_ms|0} ms</td>
                <td class="num">${r.doc_count|0}</td>
                <td>${escapeHtml(r.model || '—')}</td>
                <td>${isReadOnlyDemo() ? '<span class="conv-readonly">locked</span>' : `<button class="conv-del" onclick="deleteConversation(${r.id})">×</button>`}</td>
            </tr>`).join('');
    } catch (e) {
        console.warn('refreshConversations failed', e);
        tbody.innerHTML = `<tr><td colspan="10" style="padding:1rem; color:#b91c1c">Failed to load: ${escapeHtml(e.message)}</td></tr>`;
    }
}

async function deleteConversation(id) {
    if (isReadOnlyDemo()) {
        alert('Read-only demo: conversation deletion is disabled.');
        return;
    }
    if (!getAdminToken()) {
        alert('Admin token required. Use private operator access to enable deletes.');
        return;
    }
    if (!confirm(`Delete conversation #${id}?`)) return;
    try {
        const r = await fetch('/backend/api/conversations/' + encodeURIComponent(id), {
            method: 'DELETE',
            headers: authHeaders(),
            credentials: 'same-origin',
        });
        if (r.status === 401) { alert('Unauthorized — check admin token.'); return; }
        if (!r.ok) throw new Error('HTTP ' + r.status);
        refreshConversations();
    } catch (e) {
        alert('Delete failed: ' + e.message);
    }
}

async function clearAllConversations() {
    if (isReadOnlyDemo()) {
        alert('Read-only demo: conversation deletion is disabled.');
        return;
    }
    if (!getAdminToken()) {
        alert('Admin token required. Use private operator access to enable deletes.');
        return;
    }
    if (!confirm('Delete ALL conversations? This cannot be undone.')) return;
    try {
        const r = await fetch('/backend/api/conversations', {
            method: 'DELETE',
            headers: authHeaders(),
            credentials: 'same-origin',
        });
        if (r.status === 401) { alert('Unauthorized — check admin token.'); return; }
        if (!r.ok) throw new Error('HTTP ' + r.status);
        const data = await r.json().catch(() => ({}));
        refreshConversations();
        alert(`Deleted ${data.deleted ?? 0} conversation(s).`);
    } catch (e) {
        alert('Clear failed: ' + e.message);
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

function shortIso(iso) {
    if (!iso) return '—';
    try {
        const d = new Date(iso);
        return d.toLocaleString('da-DK', { dateStyle: 'short', timeStyle: 'short' });
    } catch (_) { return iso; }
}

// ---------- pollers ----------

async function pollSystem() {
    try {
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
    } catch (e) {
        console.warn('pollSystem failed', e);
    }
}

async function pollConnections() {
    try {
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
        const qdrantLabel = typeof qdrantPoints === 'number' ? qdrantPoints.toLocaleString('da-DK') : (qdrantPoints ?? '—');
        setText('tile-qdrant-points', qdrantLabel);
        setText('runtime-qdrant-points', qdrantLabel);
        setDot('tile-embed-dot', ['loaded', 'configured'].includes(c.embedding_model?.status) ? 'green' : 'yellow');
        const geminiColor = c.gemini?.status === 'up' ? 'green' : (c.gemini?.status === 'unconfigured' ? 'yellow' : 'red');
        setDot('tile-primary-dot', geminiColor);
        setDot('tile-fallback-dot', geminiColor);
    } catch (e) {
        console.warn('pollConnections failed', e);
    }
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
    if (!confirm('Delete ALL indexed sources? This cannot be undone.')) return;
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

async function refreshSources() {
    const tbody = document.getElementById('sources-list');
    if (!tbody) return;
    try {
        const res = await fetch('/documents', { credentials: 'same-origin' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const docs = await res.json();
        const badge = document.getElementById('source-status-badge');
        if (badge) {
            badge.textContent = `${docs.length} source${docs.length === 1 ? '' : 's'}`;
            badge.className = 'badge badge-csv';
        }
        if (docs.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; padding:2rem; color:var(--text-muted);">No indexed sources are visible yet.</td></tr>';
            return;
        }
        tbody.innerHTML = docs.map(d => {
            let badgeClass = 'badge-txt';
            if (d.type === 'PDF') badgeClass = 'badge-pdf';
            if (d.type === 'DOCX') badgeClass = 'badge-docx';
            if (d.type === 'CSV') badgeClass = 'badge-csv';
            const docKey = escapeJsString(d.filename || d.id || '');
            const docId = escapeJsString(d.id || '');
            const actions = isReadOnlyDemo()
                ? `<button onclick="viewSourceDocument('${docKey}')" class="btn btn-secondary btn-sm" title="View source">View</button><span class="locked-action">locked</span>`
                : `<button onclick="deleteSource('${docId}')" class="btn btn-danger btn-sm" title="Delete source">Delete</button><button onclick="viewSourceDocument('${docKey}')" class="btn btn-secondary btn-sm" title="View source">View</button>`;
            return `
                <tr>
                    <td data-label="File"><strong>${escapeHtml(d.filename)}</strong></td>
                    <td data-label="Type"><span class="badge ${badgeClass}">${escapeHtml(d.type)}</span></td>
                    <td data-label="Indexed">${escapeHtml(d.uploaded_at || 'N/A')}</td>
                    <td data-label="Chunks">${escapeHtml(d.chunks)}</td>
                    <td class="actions-cell">${actions}</td>
                </tr>`;
        }).join('');
    } catch (e) {
        tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color:var(--danger);">Failed to load sources: ${escapeHtml(e.message)}</td></tr>`;
    }
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
    if (!confirm('Delete this indexed source?')) return;
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

async function viewSourceDocument(key) {
    try {
        openSourceModal('Loading...', 'Fetching source...');
        let res = await fetch('/document/' + encodeURIComponent(key));
        if (!res.ok) {
            const listRes = await fetch('/documents');
            const list = await listRes.json();
            const found = list.find(d => d.filename === key);
            if (found) res = await fetch('/document/' + encodeURIComponent(found.id));
        }
        if (!res.ok) throw new Error('Source not found');
        const data = await res.json();
        openSourceModal(data.filename || key, data.text || '');
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
    try {
        const m = await jget('/backend/api/metrics');
        const memoryMb = Number(m.memory_mb || 0);
        const latencyMs = Number(m.avg_latency_ms || 0);
        const cachePct = Number(m.cache_hit_ratio_pct || 0);
        const requestsTotal = Number(m.requests_total || 0);

        setText('m-memory', memoryMb.toFixed(0));
        setText('m-latency', latencyMs.toFixed(0));
        setText('m-cache', cachePct.toFixed(0));
        setText('m-requests', requestsTotal.toLocaleString('da-DK'));
        setText('tile-cache-ratio', `${m.cache_hit_ratio_pct}%`);
        setText('tile-fallback-count', m.fallback_invocations);
        setWidth('g-memory-bar', (memoryMb / 1024) * 100);
        setWidth('g-latency-bar', (latencyMs / 3000) * 100);
        setWidth('g-cache-bar', cachePct);
        setSparkHeight('g-requests-bar', (Math.log10(requestsTotal + 1) / 3) * 100);
        setText('g-memory-meta', `${memoryMb.toFixed(0)} MB of a 1 GB soft ceiling`);
        setText('g-latency-meta', latencyMs > 3000 ? 'Above 3,000 ms target' : 'Target under 3,000 ms');
        setText('g-cache-meta', `${cachePct.toFixed(0)}% prompt/model cache reuse`);
        setText('g-requests-meta', `${requestsTotal.toLocaleString('da-DK')} handled requests`);

        const rl = m.rate_limit || {};
        const used = rl.tokens_used_today || 0;
        const budget = rl.daily_token_budget || 0;
        setText('tile-tokens-used', `${used.toLocaleString('da-DK')} / ${budget.toLocaleString('da-DK')}`);
        setText('tile-budget-reset', shortIso(rl.budget_resets_at));
        setText('g-token-readout', `${used.toLocaleString('da-DK')} / ${budget.toLocaleString('da-DK')}`);
        const budgetPct = budget > 0 ? (used / budget) * 100 : 0;
        setWidth('g-token-bar', budgetPct);
        setRing('g-token-ring', budgetPct);
        setText('g-token-meta', `${clampPct(budgetPct).toFixed(0)}% used. Resets ${shortIso(rl.budget_resets_at)}`);
        let budgetColor = 'green';
        if (budget > 0) {
            const ratio = used / budget;
            if (ratio > 0.95) budgetColor = 'red';
            else if (ratio > 0.7) budgetColor = 'yellow';
        }
        setDot('tile-budget-dot', budgetColor);
    } catch (e) {
        console.warn('pollMetrics failed', e);
    }
}

async function pollModels() {
    try {
        const data = await jget('/backend/api/models');
        MODELS_STATE = data;
        const primary = data.models.find(m => m.role === 'primary') || data.models[0];
        const fallback = data.models.find(m => m.role === 'fallback');
        setText('dd-primary-active', primary ? `${primary.label} — $${primary.input_per_m_usd}/$${primary.output_per_m_usd} per M` : 'unconfigured');
        setText('dd-fallback-active', fallback ? `${fallback.label} — $${fallback.input_per_m_usd}/$${fallback.output_per_m_usd} per M` : 'unconfigured');
        renderModelList('dd-primary-list', data.models, 'primary');
        renderModelList('dd-fallback-list', data.models, 'fallback');
    } catch (e) {
        console.warn('pollModels failed', e);
    }
}

function renderModelList(id, models, activeRole) {
    const el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = models.map(m => `
        <div class="model-row${m.role === activeRole ? ' active' : ''}">
            <div class="model-row-name">${m.label}
                ${m.role === activeRole ? '<span class="model-role">AKTIV</span>' : ''}
                <span style="font-size:0.7rem;color:#94a3b8;">(${m.vendor})</span>
            </div>
            <div class="model-row-meta">$${m.input_per_m_usd}/M input · $${m.output_per_m_usd}/M output · ${m.context_window.toLocaleString()} ctx</div>
            <div class="model-row-desc">${m.description}</div>
        </div>
    `).join('');
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
    try {
        const summary = await jget('/backend/api/usage/summary');
        document.getElementById('usage-backend').textContent = summary.backend || '—';
        const sumText = `Sidste 24t: ${summary.req_24h ?? 0} requests, ${(summary.tok_in_24h||0).toLocaleString()} input + ${(summary.tok_out_24h||0).toLocaleString()} output tokens, ${summary.errors_24h ?? 0} errors. Sidste 7d: ${summary.req_7d ?? 0} requests.`;
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
    } catch (e) {
        console.warn('pollUsage failed', e);
    }
}

function applyToChart(id, label, color, series) {
    const c = ensureChart(id, label, color);
    if (!c) return;
    c.data.labels = series.map(p => shortIso(p.ts));
    c.data.datasets[0].data = series.map(p => p.value);
    c.update('none');
}

// ---------- bootstrap ----------

async function bootstrap() {
    initSourceControls();
    await pollSystem();
    await pollModels();
    await loadPrompt();
    refreshSources();
    pollConnections();
    pollMetrics();
    pollUsage();

    setInterval(pollSystem, SYS_POLL_MS);
    setInterval(pollConnections, CONN_POLL_MS);
    setInterval(pollMetrics, METRICS_POLL_MS);
    setInterval(pollModels, SYS_POLL_MS);
    setInterval(pollUsage, USAGE_POLL_MS);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bootstrap);
} else {
    bootstrap();
}
