// Backend viewer — polling-based dashboard
// All endpoints are read-only. No write actions are possible from this page.

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

function escapeHtml(s) {
    return String(s == null ? '' : s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
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
    if (note) note.textContent = getAdminToken()
        ? 'Admin token detected — delete actions enabled.'
        : 'No admin token saved. Set one on the chat page (Knowledge Base → Admin Access Token) to enable deletes.';
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
                <td><button class="conv-del" onclick="deleteConversation(${r.id})">×</button></td>
            </tr>`).join('');
    } catch (e) {
        console.warn('refreshConversations failed', e);
        tbody.innerHTML = `<tr><td colspan="10" style="padding:1rem; color:#b91c1c">Failed to load: ${escapeHtml(e.message)}</td></tr>`;
    }
}

async function deleteConversation(id) {
    if (!getAdminToken()) {
        alert('Admin token required. Set one on the chat page (Knowledge Base → Admin Access Token).');
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
    if (!getAdminToken()) {
        alert('Admin token required. Set one on the chat page (Knowledge Base → Admin Access Token).');
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
const CHARTS = {};

// ---------- shared helpers ----------
function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value === undefined || value === null ? '—' : String(value);
}

function setDot(id, color) {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.remove('tile-dot-green', 'tile-dot-yellow', 'tile-dot-red');
    if (color) el.classList.add(`tile-dot-${color}`);
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
        setText('tile-embed-dim', `${s.embedding_dim} dim`);
        setText('tile-region', s.llm_provider);
        setText('tile-git-sha', s.git_sha);
        setText('tile-uptime', `uptime ${humanSeconds(s.uptime_seconds)}`);
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
            embedding_model: ['loaded'],
            logger: ['writing', 'stdout_only'],
            usage_db: ['up', 'memory_fallback'],
        };
        Object.entries(c).forEach(([key, val]) => {
            const span = document.querySelector(`[data-conn="${key}"]`);
            if (!span) return;
            const accepted = okStatuses[key] || ['up'];
            const good = accepted.includes(val.status);
            const dot = good ? '🟢' : '🔴';
            const label = key.replace('_', ' ');
            span.textContent = `${dot} ${label}`;
        });
        // Tile dots
        setDot('tile-qdrant-dot', c.qdrant?.status === 'up' ? 'green' : 'red');
        setText('tile-qdrant-points', c.qdrant?.points_count ?? '—');
        setDot('tile-embed-dot', c.embedding_model?.status === 'loaded' ? 'green' : 'yellow');
        const geminiColor = c.gemini?.status === 'up' ? 'green' : (c.gemini?.status === 'unconfigured' ? 'yellow' : 'red');
        setDot('tile-primary-dot', geminiColor);
        setDot('tile-fallback-dot', geminiColor);
    } catch (e) {
        console.warn('pollConnections failed', e);
    }
}

async function pollMetrics() {
    try {
        const m = await jget('/backend/api/metrics');
        setText('m-memory', (m.memory_mb || 0).toFixed(0));
        setText('m-latency', (m.avg_latency_ms || 0).toFixed(0));
        setText('m-cache', m.cache_hit_ratio_pct);
        setText('m-requests', m.requests_total);
        setText('tile-cache-ratio', `${m.cache_hit_ratio_pct}%`);
        setText('tile-fallback-count', m.fallback_invocations);

        const rl = m.rate_limit || {};
        const used = rl.tokens_used_today || 0;
        const budget = rl.daily_token_budget || 0;
        setText('tile-tokens-used', `${used.toLocaleString('da-DK')} / ${budget.toLocaleString('da-DK')}`);
        setText('tile-budget-reset', shortIso(rl.budget_resets_at));
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
    await pollSystem();
    await pollModels();
    await loadPrompt();
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
