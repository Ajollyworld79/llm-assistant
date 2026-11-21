// Helios Console.
// Public status panels poll without credentials. Operator features (sources, conversations,
// uploads, deletes) use an HttpOnly session cookie set by POST /backend/api/session; every
// request carries X-Requested-With as a CSRF guard. No token is ever stored in the browser.
(() => {
    'use strict';

    const { toast, escapeHtml, openDialog, closeDialog } = window.Helios;
    const $ = (id) => document.getElementById(id);

    const SESSION_URL = '/backend/api/session';
    const LOCKED = document.body.dataset.readOnly === 'true';
    const STATE = { authed: false };

    const POLL = { system: 30_000, connections: 5_000, metrics: 5_000, usage: 30_000, retry: 30_000 };
    const MEMORY_CEILING_MB = 1024;
    const LATENCY_TARGET_MS = 3000;
    const CONV_PAGE_SIZE = 50;

    const canMutate = () => !LOCKED && STATE.authed;
    const num = (n) => Number(n || 0).toLocaleString('en-GB');

    // ---------- tiny DOM helper ----------

    function h(tag, props, ...children) {
        const node = document.createElement(tag);
        for (const [key, value] of Object.entries(props || {})) {
            if (value === null || value === undefined || value === false) continue;
            if (key === 'class') node.className = value;
            else if (key === 'text') node.textContent = value;
            else if (key === 'on') for (const [evt, fn] of Object.entries(value)) node.addEventListener(evt, fn);
            else node.setAttribute(key, value === true ? '' : value);
        }
        for (const child of children.flat()) {
            if (child === null || child === undefined || child === false) continue;
            node.append(child);
        }
        return node;
    }

    function setText(id, value) {
        const node = $(id);
        if (node) node.textContent = value === undefined || value === null ? '—' : String(value);
    }

    function truncate(value, max) {
        const s = String(value == null ? '' : value);
        return s.length > max ? s.slice(0, max) + '…' : s;
    }

    // ---------- date formatting ----------

    const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const pad2 = (n) => String(n).padStart(2, '0');

    function fmtDateTime(iso, compact) {
        if (iso === null || iso === undefined || iso === '') return '—';
        const d = new Date(iso);
        if (Number.isNaN(d.getTime())) return String(iso);
        const day = `${pad2(d.getDate())} ${MONTHS[d.getMonth()]}`;
        const hm = `${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
        if (compact) return `${day} ${hm}`;
        return `${day} ${d.getFullYear()} ${hm}:${pad2(d.getSeconds())}`;
    }

    const fmtShort = (iso) => fmtDateTime(iso, true);

    function humanSeconds(s) {
        s = Number(s) || 0;
        if (s < 60) return `${s | 0}s`;
        if (s < 3600) return `${(s / 60) | 0}m`;
        if (s < 86400) return `${(s / 3600) | 0}h ${((s % 3600) / 60) | 0}m`;
        return `${(s / 86400) | 0}d`;
    }

    // ---------- API + operator session ----------

    async function api(path, { method = 'GET', body, headers, timeout = 20_000 } = {}) {
        const res = await fetch(path, {
            method,
            body,
            credentials: 'same-origin',
            headers: { 'X-Requested-With': 'helios', ...headers },
            signal: AbortSignal.timeout(timeout),
        });
        if (res.status === 401) setAuth(false);
        return res;
    }

    async function jget(path) {
        const res = await api(path);
        if (!res.ok) throw new Error(`${path}: ${res.status}`);
        return res.json();
    }

    function setAuth(value) {
        if (STATE.authed === value) return;
        STATE.authed = value;
        document.body.dataset.auth = String(value);
        if (value) {
            refreshSources();
            refreshConversations();
        } else {
            showSourcesSignedOut();
            CONV.items = [];
            CONV.next = null;
            CONV.error = 'unauthorized';
            renderConversations();
        }
    }

    async function checkSession() {
        try {
            const res = await fetch(SESSION_URL, { credentials: 'same-origin', headers: { 'X-Requested-With': 'helios' }, signal: AbortSignal.timeout(8000) });
            const data = res.ok ? await res.json() : {};
            setAuth(Boolean(data.authenticated));
        } catch (err) {
            console.warn('Session check failed', err);
            setAuth(false);
        }
    }

    function openLogin(opener) {
        const error = $('login-error');
        error.hidden = true;
        $('login-token').value = '';
        openDialog($('login-dialog'), opener);
    }

    async function submitLogin(e) {
        e.preventDefault();
        const input = $('login-token');
        const error = $('login-error');
        const submit = $('login-submit');
        const fail = (message) => { error.textContent = message; error.hidden = false; input.focus(); };
        if (!input.value) { fail('Enter the operator token.'); return; }
        error.hidden = true;
        submit.disabled = true;
        try {
            const res = await fetch(SESSION_URL, {
                method: 'POST',
                credentials: 'same-origin',
                headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'helios' },
                body: JSON.stringify({ token: input.value }),
                signal: AbortSignal.timeout(15_000),
            });
            if (res.ok) {
                input.value = '';
                closeDialog($('login-dialog'));
                setAuth(true);
                toast('Signed in as operator.');
            } else if (res.status === 401) {
                fail('That token was not accepted.');
            } else if (res.status === 429) {
                fail('Too many attempts. Wait a moment and try again.');
            } else {
                fail(`Sign-in failed (error ${res.status}).`);
            }
        } catch (err) {
            console.warn('Sign-in failed', err);
            fail('Could not reach the server. Try again.');
        } finally {
            submit.disabled = false;
        }
    }

    async function logout() {
        try {
            await fetch(SESSION_URL, { method: 'DELETE', credentials: 'same-origin', headers: { 'X-Requested-With': 'helios' }, signal: AbortSignal.timeout(8000) });
        } catch (err) {
            console.warn('Sign-out request failed', err);
        }
        setAuth(false);
        toast('Signed out.');
    }

    // ---------- shared UI pieces ----------

    function noteRow(colspan, text, { error = false, signIn = false } = {}) {
        const cell = h('td', { colspan: String(colspan), class: 'table-note' + (error ? ' table-note-error' : '') }, text);
        if (signIn) {
            cell.append(h('button', { type: 'button', class: 'btn btn-primary btn-sm', text: 'Sign in', on: { click: (e) => openLogin(e.currentTarget) } }));
        }
        return h('tr', null, cell);
    }

    async function copyText(text, label) {
        try {
            await navigator.clipboard.writeText(text);
            toast(`${label} copied.`);
        } catch (_) {
            const area = h('textarea', { class: 'sr-only', 'aria-hidden': 'true', tabindex: '-1' });
            area.value = text;
            document.body.appendChild(area);
            area.select();
            let ok = false;
            try { ok = document.execCommand('copy'); } catch (__) { ok = false; }
            area.remove();
            toast(ok ? `${label} copied.` : 'Copy failed.', ok);
        }
    }

    // Promise-based confirmation in a native dialog. Cancel is the default focus.
    function confirmDialog(message, okLabel = 'Delete', title = 'Are you sure?') {
        const dialog = $('confirm-dialog');
        $('confirm-title').textContent = title;
        $('confirm-msg').textContent = message;
        $('confirm-ok').textContent = okLabel;
        dialog.returnValue = ''; // Escape/backdrop close must read as "cancel"
        openDialog(dialog);
        return new Promise((resolve) => {
            dialog.addEventListener('close', () => resolve(dialog.returnValue === 'ok'), { once: true });
        });
    }

    // ---------- tabs ----------

    const TAB_NAMES = ['overview', 'sources', 'models', 'prompt', 'usage', 'conversations'];

    function setTab(name, { focus = false } = {}) {
        for (const tab of document.querySelectorAll('.bk-tab')) {
            const selected = tab.dataset.tab === name;
            tab.setAttribute('aria-selected', String(selected));
            tab.tabIndex = selected ? 0 : -1;
            if (selected && focus) tab.focus();
        }
        for (const panel of document.querySelectorAll('.bk-tab-panel')) panel.hidden = panel.id !== `tab-${name}`;
        if (name === 'usage') Object.values(CHARTS).forEach((c) => c && c.resize());
        if (name === 'sources' && STATE.authed) refreshSources();
        if (name === 'conversations' && STATE.authed) refreshConversations();
    }

    function onTabKeydown(e) {
        const tab = e.target.closest('.bk-tab');
        if (!tab) return;
        const idx = TAB_NAMES.indexOf(tab.dataset.tab);
        let next = idx;
        if (e.key === 'ArrowRight') next = (idx + 1) % TAB_NAMES.length;
        else if (e.key === 'ArrowLeft') next = (idx - 1 + TAB_NAMES.length) % TAB_NAMES.length;
        else if (e.key === 'Home') next = 0;
        else if (e.key === 'End') next = TAB_NAMES.length - 1;
        else return;
        e.preventDefault();
        setTab(TAB_NAMES[next], { focus: true });
    }

    // ---------- conversations ----------

    const CONV = { items: [], total: 0, next: null, q: '', seq: 0, loading: false, error: 'unauthorized' };

    function updateConvChrome() {
        const more = $('conv-more-btn');
        more.hidden = CONV.next === null || CONV.error !== null;
        more.disabled = CONV.loading;
        $('conv-count').textContent = CONV.error ? '' : `Showing ${CONV.items.length} of ${CONV.total}`;
    }

    function renderConversations() {
        const tbody = $('conv-tbody');
        tbody.replaceChildren();
        if (CONV.error === 'unauthorized') {
            tbody.appendChild(noteRow(9, 'Sign in as operator to read conversations.', { signIn: true }));
        } else if (CONV.error === 'unavailable') {
            tbody.appendChild(noteRow(9, 'The conversation store is unavailable.', { error: true }));
        } else if (CONV.error) {
            tbody.appendChild(noteRow(9, `Could not load conversations (${CONV.error}).`, { error: true }));
        } else if (CONV.items.length === 0) {
            tbody.appendChild(noteRow(9, CONV.loading ? 'Loading…' : (CONV.q ? 'No conversations match your search.' : 'No conversations yet.')));
        } else {
            const showDelete = canMutate();
            CONV.items.forEach((r) => tbody.appendChild(convRow(r, showDelete)));
        }
        updateConvChrome();
    }

    function convRow(r, showDelete) {
        const tr = h('tr');
        const open = (opener) => openConvDialog(r, opener);
        tr.append(
            h('td', { class: 'conv-time', title: r.ts || '' },
                h('button', { type: 'button', class: 'link-btn', text: fmtShort(r.ts), 'aria-label': `Open conversation ${r.id ?? ''} from ${fmtDateTime(r.ts)}`, on: { click: (e) => open(e.currentTarget) } })),
            h('td', { class: 'mono col-opt-sm', title: r.thread_id || '', text: (r.thread_id || '').slice(0, 8) || '—' }),
            h('td', { title: truncate(r.query, 400), text: truncate(r.query, 120) }),
            h('td', { class: 'col-opt', title: truncate(r.answer, 400), text: truncate(r.answer, 160) }),
            h('td', { class: 'num col-opt-sm', text: `${r.total_ms | 0} ms` }),
            h('td', { class: 'num col-opt', text: `${num(r.tokens_input)} / ${num(r.tokens_output)}` }),
            h('td', { class: 'num col-opt', text: String(r.doc_count | 0) }),
            h('td', { class: 'col-opt', text: r.model || '—' }),
        );
        const actions = h('td', { class: 'actions-cell' });
        if (showDelete) {
            actions.appendChild(h('button', {
                type: 'button', class: 'btn btn-danger btn-sm', text: 'Delete',
                'aria-label': `Delete conversation ${r.id ?? ''}`,
                on: { click: () => deleteConversation(r.id) },
            }));
        }
        tr.appendChild(actions);
        return tr;
    }

    async function fetchConversationsPage(beforeId) {
        const params = new URLSearchParams({ limit: String(CONV_PAGE_SIZE) });
        if (beforeId !== null && beforeId !== undefined) params.set('before_id', String(beforeId));
        if (CONV.q) params.set('q', CONV.q);
        const res = await api('/backend/api/conversations?' + params);
        if (res.status === 401) return { error: 'unauthorized' };
        if (res.status === 500) return { error: 'unavailable' };
        if (!res.ok) return { error: `error ${res.status}` };
        return { data: await res.json() };
    }

    async function loadConversations(append) {
        if (!STATE.authed) return;
        const seq = ++CONV.seq;
        CONV.loading = true;
        if (!append) { CONV.items = []; CONV.next = null; CONV.error = null; }
        renderConversations();
        let res;
        try {
            res = await fetchConversationsPage(append ? CONV.next : null);
        } catch (err) {
            console.warn('Loading conversations failed', err);
            res = { error: 'network error' };
        }
        if (seq !== CONV.seq) return; // superseded by a newer request
        CONV.loading = false;
        if (res.error) {
            Object.assign(CONV, { error: res.error, items: [], next: null, total: 0 });
        } else {
            const { data } = res;
            CONV.error = null;
            setText('conv-backend', data.backend || '—');
            CONV.items = append ? CONV.items.concat(data.items || []) : (data.items || []);
            CONV.total = data.total | 0;
            CONV.next = data.next_before_id === undefined ? null : data.next_before_id;
        }
        renderConversations();
    }

    function refreshConversations() {
        CONV.q = $('conv-search').value.trim();
        return loadConversations(false);
    }

    function openConvDialog(r, opener) {
        const body = $('conv-dialog-body');
        body.replaceChildren();
        $('conv-dialog-title').textContent = `Conversation #${r.id ?? ''}`;

        const meta = (label, value) => h('div', null, h('dt', { text: label }), h('dd', null, value));
        const thread = r.thread_id
            ? h('button', {
                type: 'button', class: 'link-btn', text: r.thread_id, title: 'Show only this thread',
                on: { click: () => {
                    $('conv-search').value = r.thread_id;
                    closeDialog($('conv-dialog'));
                    refreshConversations();
                } },
            })
            : '—';
        body.appendChild(h('dl', { class: 'conv-detail-meta' },
            meta('Time', fmtDateTime(r.ts)),
            meta('Thread', thread),
            meta('Model', r.model || '—'),
            meta('Embedding', `${r.embed_ms | 0} ms`),
            meta('Answer', `${r.ai_ms | 0} ms`),
            meta('Total', `${r.total_ms | 0} ms`),
            meta('Tokens in / out', `${num(r.tokens_input)} / ${num(r.tokens_output)}`),
            meta('Documents', String(r.doc_count | 0)),
        ));
        const block = (title, text) => h('section', { class: 'conv-detail-block' },
            h('h3', null, title, h('button', { type: 'button', class: 'btn btn-secondary btn-sm', text: 'Copy', 'aria-label': `Copy ${title.toLowerCase()}`, on: { click: () => copyText(text, title) } })),
            h('pre', { class: 'conv-detail-text', text: text || '—' }));
        body.append(block('Question', r.query || ''), block('Answer', r.answer || ''));
        openDialog($('conv-dialog'), opener);
    }

    async function deleteConversation(id) {
        if (!canMutate()) return;
        if (!await confirmDialog(`Delete conversation #${id}?`, 'Delete')) return;
        try {
            const res = await api('/backend/api/conversations/' + encodeURIComponent(id), { method: 'DELETE' });
            if (res.status === 401) { toast('Your session has expired. Sign in again.', false); return; }
            if (!res.ok) throw new Error(`error ${res.status}`);
            toast('Conversation deleted.');
            refreshConversations();
        } catch (err) {
            toast(`Delete failed: ${err.message}`, false);
        }
    }

    async function clearAllConversations() {
        if (!canMutate()) return;
        if (!await confirmDialog('Delete ALL conversations? This cannot be undone.', 'Delete all')) return;
        try {
            const res = await api('/backend/api/conversations', { method: 'DELETE' });
            if (res.status === 401) { toast('Your session has expired. Sign in again.', false); return; }
            if (!res.ok) throw new Error(`error ${res.status}`);
            const data = await res.json().catch(() => ({}));
            toast(`Deleted ${data.deleted ?? 0} conversation(s).`);
            refreshConversations();
        } catch (err) {
            toast(`Delete failed: ${err.message}`, false);
        }
    }

    // ---------- status tiles, metrics, models ----------

    const DOT_LABEL = { green: 'Healthy', yellow: 'Needs attention', red: 'Down' };

    function setDot(id, color) {
        const node = $(id);
        if (!node) return;
        node.className = 'tile-dot dot ' + (color === 'green' ? 'dot-ok' : color === 'yellow' ? 'dot-warn' : 'dot-bad');
        node.setAttribute('aria-label', DOT_LABEL[color] || 'Status unknown');
    }

    function setBar(id, pct) {
        const node = $(id);
        if (node) node.value = Math.max(0, Math.min(100, Number(pct) || 0));
    }

    async function pollSystem() {
        const s = await jget('/backend/api/system');
        setText('tile-primary-model', s.model_primary);
        setText('tile-fallback-model', s.model_fallback);
        setText('tile-embed-model', s.embedding_model);
        setText('tile-embed-dim', `${s.embedding_provider} · ${s.embedding_dim} dimensions`);
        setText('tile-region', s.llm_provider);
        const build = [s.git_sha ? `build ${String(s.git_sha).slice(0, 7)}` : null, s.uptime_seconds != null ? `up ${humanSeconds(s.uptime_seconds)}` : null].filter(Boolean);
        setText('build-info', build.length ? build.join(' · ') : '—');
        setText('runtime-provider', s.llm_provider);
        setText('runtime-embed', truncate(s.embedding_model, 28));
        setText('runtime-primary', truncate(s.model_primary, 28));
        const pre = $('config-pre');
        if (pre) pre.textContent = JSON.stringify(s.config, null, 2);
    }

    const CONN_OK = {
        qdrant: ['up'], gemini: ['up'], embedding_model: ['loaded', 'configured'],
        logger: ['writing', 'stdout_only'], usage_db: ['up', 'memory_fallback'], chat_db: ['up', 'memory_fallback'],
    };

    async function pollConnections() {
        const c = await jget('/backend/api/connections');
        for (const [key, val] of Object.entries(c)) {
            const chip = document.querySelector(`[data-conn="${key}"]`);
            if (!chip) continue;
            const good = (CONN_OK[key] || ['up']).includes(val.status);
            const warn = !good && val.status === 'unconfigured';
            chip.querySelector('.dot').className = 'dot ' + (good ? 'dot-ok' : warn ? 'dot-warn' : 'dot-bad');
            chip.dataset.base = chip.dataset.base || chip.querySelector('.chip-label').textContent;
            const suffix = (key === 'chat_db' && val.count != null) ? ` (${val.count})` : '';
            chip.querySelector('.chip-label').textContent = chip.dataset.base + suffix;
            chip.querySelector('.chip-state').textContent = good ? ': up' : warn ? ': not configured' : ': down';
        }
        setDot('tile-qdrant-dot', c.qdrant?.status === 'up' ? 'green' : 'red');
        const points = c.qdrant?.points_count;
        const label = typeof points === 'number' ? points.toLocaleString('en-GB') : (points ?? '—');
        setText('tile-qdrant-points', label);
        setText('runtime-qdrant-points', label);
        setDot('tile-embed-dot', ['loaded', 'configured'].includes(c.embedding_model?.status) ? 'green' : 'yellow');
        const gemini = c.gemini?.status === 'up' ? 'green' : (c.gemini?.status === 'unconfigured' ? 'yellow' : 'red');
        setDot('tile-primary-dot', gemini);
        setDot('tile-fallback-dot', gemini);
    }

    async function pollMetrics() {
        const m = await jget('/backend/api/metrics');
        const memory = Number(m.memory_mb || 0);
        const latency = Number(m.avg_latency_ms || 0);
        const cache = Number(m.cache_hit_ratio_pct || 0);
        const requests = Number(m.requests_total || 0);

        setText('m-memory', memory.toFixed(0));
        setText('m-latency', latency.toFixed(0));
        setText('m-cache', cache.toFixed(0));
        setText('m-requests', requests.toLocaleString('en-GB'));
        setText('tile-cache-ratio', `${cache.toFixed(0)}%`);
        setText('tile-fallback-count', m.fallback_invocations);
        setBar('g-memory-bar', (memory / MEMORY_CEILING_MB) * 100);
        setBar('g-latency-bar', (latency / LATENCY_TARGET_MS) * 100);
        setBar('g-cache-bar', cache);
        setText('g-memory-meta', `${memory.toFixed(0)} MB of a 1 GB soft ceiling`);
        setText('g-latency-meta', latency > LATENCY_TARGET_MS ? 'Above the 3,000 ms target' : 'Target under 3,000 ms');
        setText('g-cache-meta', `${cache.toFixed(0)}% of prompts served from cache`);

        const rl = m.rate_limit || {};
        const used = rl.tokens_used_today || 0;
        const budget = rl.daily_token_budget || 0;
        const pct = budget > 0 ? (used / budget) * 100 : 0;
        setText('g-token-readout', `${num(used)} / ${num(budget)}`);
        setBar('g-token-bar', pct);
        setText('g-token-meta', `${Math.min(100, pct).toFixed(0)}% used. Resets ${fmtShort(rl.budget_resets_at)}`);
    }

    async function pollModels() {
        const data = await jget('/backend/api/models');
        const describe = (m) => (m ? `${m.label} — $${m.input_per_m_usd} / $${m.output_per_m_usd} per M tokens` : 'Not configured');
        setText('dd-primary-active', describe(data.models.find((m) => m.role === 'primary') || data.models[0]));
        setText('dd-fallback-active', describe(data.models.find((m) => m.role === 'fallback')));
        renderModelList('dd-primary-list', data.models, 'primary');
        renderModelList('dd-fallback-list', data.models, 'fallback');
    }

    function renderModelList(id, models, activeRole) {
        $(id).replaceChildren(...models.map((m) => {
            const active = m.role === activeRole;
            const ctx = Number(m.context_window);
            return h('div', { class: 'model-row' + (active ? ' active' : '') },
                h('div', { class: 'model-row-name' }, m.label || '—',
                    active ? h('span', { class: 'model-role', text: 'Active' }) : null,
                    h('span', { class: 'model-vendor', text: m.vendor || '—' })),
                h('div', { class: 'model-row-meta', text: `$${m.input_per_m_usd}/M input · $${m.output_per_m_usd}/M output` + (Number.isFinite(ctx) ? ` · ${ctx.toLocaleString('en-GB')} context` : '') }),
                m.description ? h('div', { class: 'model-row-desc', text: m.description }) : null);
        }));
    }

    function toggleDropdown(which, button) {
        const list = $(`dd-${which}-list`);
        list.hidden = !list.hidden;
        button.setAttribute('aria-expanded', String(!list.hidden));
    }

    // ---------- sources ----------

    let sourceFiles = [];

    function showSourcesSignedOut() {
        const tbody = $('sources-list');
        if (!tbody) return;
        tbody.replaceChildren(noteRow(5, 'Sign in as operator to view indexed sources.', { signIn: true }));
        const badge = $('source-status-badge');
        badge.textContent = 'Sign in to view';
        badge.className = 'badge';
    }

    function sourceRow(d) {
        const typeBadge = { PDF: 'badge-danger', DOCX: 'badge-accent', CSV: 'badge-success' }[d.type] || '';
        const view = h('button', { type: 'button', class: 'btn btn-secondary btn-sm', text: 'View', 'aria-label': `View ${d.filename || d.id}`, on: { click: (e) => viewSource(d.id, d.filename, e.currentTarget) } });
        const actions = h('td', { class: 'actions-cell' }, view);
        if (canMutate()) {
            actions.append(h('button', { type: 'button', class: 'btn btn-danger btn-sm', text: 'Delete', 'aria-label': `Delete ${d.filename || d.id}`, on: { click: () => deleteSource(d.id) } }));
        }
        return h('tr', null,
            h('td', { 'data-label': 'File' }, h('strong', { text: d.filename || d.id || '—' })),
            h('td', { 'data-label': 'Type' }, h('span', { class: 'badge ' + typeBadge, text: d.type || '—' })),
            h('td', { 'data-label': 'Indexed', text: d.uploaded_at ? fmtDateTime(d.uploaded_at) : 'N/A' }),
            h('td', { class: 'num', 'data-label': 'Chunks', text: d.chunks === undefined || d.chunks === null ? '—' : String(d.chunks) }),
            actions);
    }

    async function refreshSources() {
        if (!STATE.authed) { showSourcesSignedOut(); return; }
        const tbody = $('sources-list');
        const badge = $('source-status-badge');
        try {
            const res = await api('/documents');
            if (res.status === 401) return; // setAuth(false) already rendered the sign-in prompt
            if (!res.ok) throw new Error(`error ${res.status}`);
            const docs = await res.json();
            badge.textContent = `${docs.length} source${docs.length === 1 ? '' : 's'}`;
            badge.className = 'badge badge-success';
            tbody.replaceChildren(...(docs.length ? docs.map(sourceRow) : [noteRow(5, 'No sources have been indexed yet.')]));
        } catch (err) {
            console.warn('Loading sources failed', err);
            badge.textContent = 'Error';
            badge.className = 'badge badge-danger';
            tbody.replaceChildren(noteRow(5, `Could not load sources: ${err.message}`, { error: true }));
        }
    }

    function addSourcesToQueue(files) {
        sourceFiles = files;
        $('source-file-list').innerHTML = files.map((f) => `<li>${escapeHtml(f.name)} (${(f.size / 1024).toFixed(1)} KB)</li>`).join('');
        $('source-upload-queue').hidden = files.length === 0;
        $('source-upload-progress').textContent = '';
    }

    async function uploadSources() {
        if (!canMutate() || sourceFiles.length === 0) return;
        const btn = $('source-upload-btn');
        const progress = $('source-upload-progress');
        btn.disabled = true;
        const failures = [];
        for (let i = 0; i < sourceFiles.length; i++) {
            const file = sourceFiles[i];
            progress.textContent = `Uploading ${i + 1} of ${sourceFiles.length}: ${file.name}`;
            const form = new FormData();
            form.append('file', file);
            try {
                const res = await api('/upload', { method: 'POST', body: form, timeout: 180_000 });
                if (res.status === 401) { failures.push(`${file.name}: session expired`); break; }
                if (!res.ok) {
                    const data = await res.json().catch(() => ({}));
                    failures.push(`${file.name}: ${data.error || `error ${res.status}`}`);
                }
            } catch (err) {
                failures.push(`${file.name}: ${err.message}`);
            }
        }
        btn.disabled = false;
        const total = sourceFiles.length;
        sourceFiles = [];
        $('source-file-input').value = '';
        $('source-upload-queue').hidden = true;
        progress.textContent = failures.length ? `Failed: ${failures.join('; ')}` : '';
        if (failures.length) toast(`${failures.length} of ${total} upload(s) failed.`, false);
        else toast(total === 1 ? 'Source uploaded.' : `${total} sources uploaded.`);
        refreshSources();
    }

    async function clearSources() {
        if (!canMutate()) return;
        if (!await confirmDialog('Delete ALL indexed sources? This cannot be undone.', 'Delete all')) return;
        try {
            const res = await api('/reset', { method: 'POST' });
            const data = await res.json().catch(() => ({}));
            if (res.status === 401) { toast('Your session has expired. Sign in again.', false); return; }
            if (!res.ok) throw new Error(data.error || `error ${res.status}`);
            toast(data.deleted !== undefined ? `Cleared ${data.deleted} points.` : 'Sources cleared.');
            refreshSources();
        } catch (err) {
            toast(`Clear failed: ${err.message}`, false);
        }
    }

    async function optimizeSources(button) {
        if (!canMutate()) return;
        const label = button.textContent;
        button.disabled = true;
        button.textContent = 'Optimizing…';
        try {
            const res = await api('/optimize', { method: 'POST' });
            const data = await res.json().catch(() => ({}));
            if (res.status === 401) { toast('Your session has expired. Sign in again.', false); return; }
            if (!res.ok) throw new Error(data.error || `error ${res.status}`);
            toast(data.message || 'Optimization started.');
        } catch (err) {
            toast(`Optimize failed: ${err.message}`, false);
        } finally {
            button.disabled = false;
            button.textContent = label;
        }
    }

    async function deleteSource(id) {
        if (!canMutate()) return;
        if (!await confirmDialog('Delete this indexed source?', 'Delete')) return;
        try {
            const res = await api('/document/' + encodeURIComponent(id), { method: 'DELETE' });
            if (res.status === 401) { toast('Your session has expired. Sign in again.', false); return; }
            if (!res.ok) throw new Error(`error ${res.status}`);
            toast('Source deleted.');
            refreshSources();
        } catch (err) {
            toast(`Delete failed: ${err.message}`, false);
        }
    }

    function showSourceDialog(title, text, opener) {
        $('source-dialog-title').textContent = title;
        $('source-dialog-text').textContent = text;
        openDialog($('source-dialog'), opener);
    }

    async function viewSource(id, filename, opener) {
        showSourceDialog(filename || id || 'Source', 'Loading…', opener);
        try {
            let res = await api('/document/' + encodeURIComponent(id || filename));
            if (!res.ok && filename && id !== filename) res = await api('/document/' + encodeURIComponent(filename));
            if (res.status === 401) { closeDialog($('source-dialog')); return; }
            if (!res.ok) throw new Error('Source not found');
            const data = await res.json();
            showSourceDialog(data.filename || filename || id, data.text || '', opener);
        } catch (err) {
            $('source-dialog-text').textContent = `Could not load this source. ${err.message}`;
        }
    }

    // ---------- prompt ----------

    let promptState = null;

    async function loadPrompt() {
        try {
            promptState = await jget('/backend/api/prompt');
            const area = $('prompt-system');
            if (!area.value) area.value = promptState.system_prompt;
            updatePreview();
        } catch (err) {
            console.warn('Loading the prompt failed', err);
            $('prompt-preview').textContent = 'The prompt is not available right now.';
        }
    }

    function updatePreview() {
        if (!promptState) return;
        const system = $('prompt-system').value;
        const user = `Konteksten:\n${promptState.example_context}\n\nSpørgsmål: ${promptState.example_query}`;
        $('prompt-preview').textContent =
            `system:\n${system.slice(0, 500)}${system.length > 500 ? '…' : ''}\n\nuser:\n${user}\n\n(JSON schema: ${promptState.schema.name}, strict=${promptState.schema.strict})`;
    }

    function importPrompt(e) {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = () => { $('prompt-system').value = reader.result; updatePreview(); };
        reader.readAsText(file);
        e.target.value = '';
    }

    function exportPrompt() {
        const url = URL.createObjectURL(new Blob([$('prompt-system').value], { type: 'text/plain' }));
        const link = h('a', { href: url, download: 'system-prompt.txt' });
        link.click();
        URL.revokeObjectURL(url);
    }

    // ---------- usage charts (colours follow the CSS tokens, light and dark) ----------

    const CHARTS = {};
    const CHART_SERIES = {
        'chart-requests': { label: 'Requests', token: '--chart-1' },
        'chart-tokens': { label: 'Tokens', token: '--chart-2' },
        'chart-cache': { label: 'Cache hit %', token: '--chart-3' },
        'chart-fallback': { label: 'Fallback %', token: '--chart-4' },
    };
    const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

    function styleChart(chart, token) {
        const color = cssVar(token);
        const dataset = chart.data.datasets[0];
        dataset.borderColor = color;
        dataset.backgroundColor = color + '33';
        for (const axis of ['x', 'y']) {
            chart.options.scales[axis].ticks.color = cssVar('--ink-mute');
            chart.options.scales[axis].grid.color = cssVar('--rule');
        }
    }

    function ensureChart(id) {
        if (CHARTS[id]) return CHARTS[id];
        const canvas = $(id);
        if (!canvas || !window.Chart) return null;
        const { label, token } = CHART_SERIES[id];
        window.Chart.defaults.font.family = cssVar('--sans');
        CHARTS[id] = new window.Chart(canvas, {
            type: 'line',
            data: { labels: [], datasets: [{ label, data: [], fill: true, tension: 0.25, pointRadius: 0 }] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 6 }, grid: {} },
                    y: { ticks: {}, grid: {}, beginAtZero: true },
                },
            },
        });
        styleChart(CHARTS[id], token);
        CHARTS[id].update('none');
        return CHARTS[id];
    }

    function applyToChart(id, series) {
        const chart = ensureChart(id);
        if (!chart) return;
        const wrap = chart.canvas.parentElement;
        const empty = wrap.querySelector('.chart-empty');
        const points = series || [];
        if (points.length === 0) {
            if (!empty) wrap.appendChild(h('div', { class: 'chart-empty', text: 'No data for this period yet.' }));
        } else if (empty) {
            empty.remove();
        }
        chart.data.labels = points.map((p) => fmtShort(p.ts));
        chart.data.datasets[0].data = points.map((p) => p.value);
        chart.update('none');
    }

    async function pollUsage() {
        const s = await jget('/backend/api/usage/summary');
        setText('usage-backend', s.backend);
        $('usage-summary').textContent =
            `Last 24 h: ${num(s.req_24h)} requests, ${num(s.tok_in_24h)} input and ${num(s.tok_out_24h)} output tokens, ${num(s.errors_24h)} errors. Last 7 days: ${num(s.req_7d)} requests.`;
        const [req, tok, cache, fallback] = await Promise.all([
            jget('/backend/api/usage/timeseries?metric=requests&range=24h'),
            jget('/backend/api/usage/timeseries?metric=tokens&range=24h'),
            jget('/backend/api/usage/timeseries?metric=cache_hit_ratio&range=7d'),
            jget('/backend/api/usage/timeseries?metric=fallback_rate&range=7d'),
        ]);
        applyToChart('chart-requests', req.series);
        applyToChart('chart-tokens', tok.series);
        applyToChart('chart-cache', cache.series);
        applyToChart('chart-fallback', fallback.series);
    }

    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
        for (const [id, chart] of Object.entries(CHARTS)) {
            styleChart(chart, CHART_SERIES[id].token);
            chart.update('none');
        }
    });

    // ---------- poll scheduler ----------
    // One tick loop: skips while the tab is hidden, never overlaps a poller with itself,
    // and shows a "stale since" chip when a poller keeps failing.

    const POLLERS = [
        { name: 'system', fn: pollSystem, every: POLL.system },
        { name: 'connections', fn: pollConnections, every: POLL.connections },
        { name: 'metrics', fn: pollMetrics, every: POLL.metrics },
        { name: 'usage', fn: pollUsage, every: POLL.usage },
        { name: 'models', fn: pollModels, every: Infinity }, // once, retried only after a failure
    ].map((p) => ({ ...p, running: false, last: 0, failedSince: null }));

    function updateStaleIndicator() {
        const chip = $('stale-indicator');
        const failing = POLLERS.filter((p) => p.failedSince);
        if (failing.length === 0) { chip.hidden = true; return; }
        const since = new Date(Math.min(...failing.map((p) => p.failedSince.getTime())));
        chip.textContent = `Data may be stale since ${pad2(since.getHours())}:${pad2(since.getMinutes())}`;
        chip.title = `Failing: ${failing.map((p) => p.name).join(', ')}`;
        chip.hidden = false;
    }

    async function runPoller(p) {
        if (p.running) return;
        p.running = true;
        p.last = Date.now();
        try {
            await p.fn();
            p.failedSince = null;
        } catch (err) {
            console.warn(`Polling ${p.name} failed`, err);
            if (!p.failedSince) p.failedSince = new Date();
        } finally {
            p.running = false;
            updateStaleIndicator();
        }
    }

    function pollTick() {
        if (document.hidden) return;
        const now = Date.now();
        for (const p of POLLERS) {
            const interval = p.failedSince ? Math.min(p.every, POLL.retry) : p.every;
            if (p.last === 0 || now - p.last >= interval) runPoller(p);
        }
    }

    // ---------- wiring ----------

    const ACTIONS = {
        'open-login': (e, el) => openLogin(el),
        'logout': () => logout(),
        'optimize-sources': (e, el) => optimizeSources(el),
        'clear-sources': () => clearSources(),
        'refresh-sources': () => refreshSources(),
        'pick-source-file': () => $('source-file-input').click(),
        'upload-sources': () => uploadSources(),
        'toggle-dropdown': (e, el) => toggleDropdown(el.dataset.dropdown, el),
        'pick-prompt-file': () => $('prompt-import').click(),
        'export-prompt': () => exportPrompt(),
        'update-preview': () => updatePreview(),
        'refresh-conversations': () => refreshConversations(),
        'clear-conversations': () => clearAllConversations(),
        'load-more-conversations': () => { if (CONV.next !== null && !CONV.loading) loadConversations(true); },
    };

    function initSourceDropZone() {
        const zone = $('source-drop-zone');
        const input = $('source-file-input');
        if (!zone || !input) return;
        input.addEventListener('change', (e) => { addSourcesToQueue(Array.from(e.target.files || [])); e.target.value = ''; });
        zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
        zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('dragover');
            if (e.dataTransfer && e.dataTransfer.files.length) addSourcesToQueue(Array.from(e.dataTransfer.files));
        });
    }

    function bootstrap() {
        document.addEventListener('click', (e) => {
            const tab = e.target.closest('.bk-tab');
            if (tab) { setTab(tab.dataset.tab); return; }
            const el = e.target.closest('[data-action]');
            const fn = el && ACTIONS[el.dataset.action];
            if (fn) fn(e, el);
        });
        document.querySelector('.bk-tabs').addEventListener('keydown', onTabKeydown);
        $('login-form').addEventListener('submit', submitLogin);
        $('prompt-import').addEventListener('change', importPrompt);
        $('prompt-system').addEventListener('input', updatePreview);

        let searchTimer = null;
        const search = $('conv-search');
        search.addEventListener('input', () => { clearTimeout(searchTimer); searchTimer = setTimeout(refreshConversations, 300); });
        search.addEventListener('keydown', (e) => { if (e.key === 'Enter') { clearTimeout(searchTimer); refreshConversations(); } });

        initSourceDropZone();
        showSourcesSignedOut();
        renderConversations();
        checkSession();
        loadPrompt();
        pollTick();
        setInterval(pollTick, 1000);
        document.addEventListener('visibilitychange', () => { if (!document.hidden) pollTick(); });
    }

    bootstrap();
})();
