// Helios chat page.
(() => {
    'use strict';

    const { escapeHtml, toast, storage, mobile, openDialog, isApple } = window.Helios;
    const $ = (id) => document.getElementById(id);

    const THREAD_KEY = 'chat_thread_id';
    const IMAGE_TYPES = { png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', webp: 'image/webp' };
    const MAX_IMAGES = 4;
    const MAX_IMAGE_BYTES = 4 * 1024 * 1024;
    const HEALTH_POLL_MS = 30_000;

    const BOT_AVATAR_SVG = '<svg fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">' +
        '<circle cx="12" cy="12" r="4" stroke-width="2"/>' +
        '<path stroke-linecap="round" stroke-width="2" d="M12 2v3M12 19v3M2 12h3M19 12h3M4.93 4.93l2.12 2.12M16.95 16.95l2.12 2.12M4.93 19.07l2.12-2.12M16.95 7.05l2.12-2.12"/></svg>';
    const CLOCK_SVG = '<svg fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">' +
        '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>';
    const CLOSE_SVG = '<svg fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path stroke-linecap="round" stroke-width="2" d="M6 6l12 12M18 6L6 18"/></svg>';

    const history = $('chat-history');
    const thread = $('chat-thread');
    const input = $('query-input');
    const sendBtn = $('send-btn');
    const sidebar = $('sidebar');
    const backdrop = $('sidebar-backdrop');
    const menuToggle = $('menu-toggle');

    let busy = false;
    let attachments = [];

    // ---------- markdown (marked + DOMPurify) ----------

    const SANITIZE = { FORBID_TAGS: ['img', 'style', 'form', 'input', 'button'], ALLOW_DATA_ATTR: false };

    if (window.DOMPurify) {
        window.DOMPurify.addHook('afterSanitizeAttributes', (node) => {
            if (node.tagName === 'A' && node.hasAttribute('href')) {
                node.setAttribute('target', '_blank');
                node.setAttribute('rel', 'noopener noreferrer');
            }
        });
    }

    function renderMarkdown(text) {
        if (!window.marked || !window.DOMPurify) return escapeHtml(text);
        try {
            return window.DOMPurify.sanitize(window.marked.parse(String(text), { breaks: true, gfm: true }), SANITIZE);
        } catch (err) {
            console.warn('Markdown rendering failed; showing plain text.', err);
            return escapeHtml(text);
        }
    }

    // ---------- small DOM helpers ----------

    function el(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
    }

    function scrollToEnd() {
        history.scrollTop = history.scrollHeight;
    }

    function getThreadId() {
        let id = storage.get(THREAD_KEY);
        if (!id) {
            id = (window.crypto && crypto.randomUUID)
                ? crypto.randomUUID().replace(/-/g, '').slice(0, 16)
                : Date.now().toString(36) + Math.random().toString(36).slice(2, 10);
            storage.set(THREAD_KEY, id);
        }
        return id;
    }

    // ---------- sidebar (off-canvas on small screens) ----------

    function setSidebarOpen(open, { returnFocus = false } = {}) {
        const isOpen = open && mobile.matches;
        sidebar.classList.toggle('open', isOpen);
        backdrop.classList.toggle('show', isOpen);
        sidebar.inert = mobile.matches && !isOpen;
        menuToggle.setAttribute('aria-expanded', String(isOpen));
        menuToggle.setAttribute('aria-label', isOpen ? 'Close menu' : 'Open menu');
        document.body.classList.toggle('no-scroll', isOpen);
        if (isOpen) sidebar.querySelector('a, button')?.focus();
        else if (returnFocus) menuToggle.focus();
    }

    function sidebarIsOpen() {
        return sidebar.classList.contains('open');
    }

    // ---------- health ----------

    function setStatus(key, text) {
        const node = document.querySelector(`[data-status="${key}"]`);
        if (node) node.textContent = text;
    }

    function setHealthState(state, label) {
        for (const id of ['health-dot', 'header-health-dot']) {
            const dot = $(id);
            dot.className = 'dot ' + (state === 'ok' ? 'dot-ok' : state === 'warn' ? 'dot-warn' : 'dot-bad');
        }
        $('header-health-text').textContent = label;
    }

    async function checkHealth() {
        try {
            const res = await fetch('/health', { signal: AbortSignal.timeout(8000) });
            const data = await res.json();
            const parts = data.components || {};
            const live = (parts.gemini || {}).status === 'healthy';
            const qdrant = (parts.qdrant || {}).status === 'healthy';
            const points = (parts.qdrant || {}).points_count;
            setStatus('engine', live ? 'Live model' : 'Simulated');
            setStatus('qdrant', qdrant ? 'Qdrant' + (points != null ? ` · ${points} chunks` : '') : 'In-memory');
            const healthy = data.status === 'healthy';
            setStatus('overall', healthy ? 'Operational' : 'Degraded');
            setHealthState(healthy ? 'ok' : 'warn', healthy ? 'All systems operational' : 'Degraded');
        } catch (err) {
            console.warn('Health check failed', err);
            setStatus('overall', 'Offline');
            setHealthState('bad', 'Unreachable');
        }
    }

    // ---------- attachments (images only; documents are ingested in the console) ----------

    function extOf(name) {
        return (name.split('.').pop() || '').toLowerCase();
    }

    function addAttachments(files) {
        const images = files.filter((f) => IMAGE_TYPES[extOf(f.name)]);
        if (images.length !== files.length) toast('Only PNG, JPEG and WebP images can be attached.', false);
        for (const f of images) {
            if (f.size > MAX_IMAGE_BYTES) { toast(`${f.name} is larger than 4 MB.`, false); continue; }
            if (attachments.length >= MAX_IMAGES) { toast(`You can attach up to ${MAX_IMAGES} images.`, false); break; }
            attachments.push(f);
        }
        renderAttachments();
    }

    function renderAttachments() {
        const wrap = $('chat-attachments');
        wrap.hidden = attachments.length === 0;
        wrap.replaceChildren(...attachments.map((file, idx) => {
            const chip = el('span', 'chat-attachment-chip');
            chip.setAttribute('role', 'listitem');
            chip.appendChild(el('span', '', file.name));
            const remove = el('button');
            remove.type = 'button';
            remove.setAttribute('aria-label', `Remove ${file.name}`);
            remove.innerHTML = CLOSE_SVG;
            remove.addEventListener('click', () => {
                attachments.splice(idx, 1);
                renderAttachments();
                input.focus();
            });
            chip.appendChild(remove);
            return chip;
        }));
    }

    function fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                const result = String(reader.result || '');
                const comma = result.indexOf(',');
                resolve(comma >= 0 ? result.slice(comma + 1) : result);
            };
            reader.onerror = () => reject(reader.error);
            reader.readAsDataURL(file);
        });
    }

    // ---------- messages ----------

    function hideHero() {
        const hero = $('welcome-hero');
        if (hero) hero.hidden = true;
    }

    function showTyping() {
        removeTyping();
        const row = el('div', 'message bot');
        row.id = 'typing-message';
        const avatar = el('div', 'msg-avatar');
        avatar.setAttribute('aria-hidden', 'true');
        avatar.innerHTML = BOT_AVATAR_SVG;
        const main = el('div', 'msg-main');
        const indicator = el('div', 'typing-indicator');
        indicator.setAttribute('role', 'status');
        indicator.append(el('i'), el('i'), el('i'), el('span', 'sr-only', 'Helios is thinking…'));
        main.append(el('div', 'message-label', 'Helios'), indicator);
        row.append(avatar, main);
        thread.appendChild(row);
        scrollToEnd();
    }

    function removeTyping() {
        $('typing-message')?.remove();
    }

    function buildResultCard(result, allResults) {
        const filename = String(result.filename == null ? '' : result.filename);
        const score = Number(result.score) || 0;
        const card = el('button', 'result-card');
        card.type = 'button';
        card.setAttribute('aria-haspopup', 'dialog');
        card.setAttribute('aria-label', `Open excerpt from ${filename}, ${(score * 100).toFixed(0)}% match`);

        const header = el('span', 'result-header');
        const badge = el('span', 'badge ' + (score > 0.8 ? 'badge-success' : score > 0.6 ? 'badge-warning' : ''), `${(score * 100).toFixed(0)}%`);
        header.append(el('span', '', filename), badge);
        card.append(header, el('span', 'result-text', result.chunk_text == null ? '' : String(result.chunk_text)));
        card.addEventListener('click', () => showExcerpts(filename, allResults, card));
        return card;
    }

    function addMessage(type, text, { results = null, meta = null, error = false } = {}) {
        hideHero();
        const row = el('div', `message ${type}${error ? ' is-error' : ''}`);
        if (type === 'bot') {
            const avatar = el('div', 'msg-avatar');
            avatar.setAttribute('aria-hidden', 'true');
            avatar.innerHTML = BOT_AVATAR_SVG;
            row.appendChild(avatar);
        }

        const main = el('div', 'msg-main');
        main.appendChild(el('div', 'message-label', type === 'user' ? 'You' : 'Helios'));

        const bubble = el('div', 'msg-bubble');
        if (type === 'user' || error) bubble.textContent = text;
        else bubble.innerHTML = renderMarkdown(text);
        main.appendChild(bubble);

        if (results && results.length > 0) {
            main.appendChild(el('p', 'results-label', `Sources (${results.length})`));
            const grid = el('div', 'results-grid');
            results.forEach((r) => grid.appendChild(buildResultCard(r, results)));
            main.appendChild(grid);
        }

        if (type === 'bot' && meta && meta.totalMs != null) {
            const line = el('div', 'msg-meta');
            const time = el('span');
            time.innerHTML = CLOCK_SVG;
            time.append(`${(meta.totalMs / 1000).toFixed(1)}s`);
            line.appendChild(time);
            if (meta.sources) line.append('·', el('span', '', `${meta.sources} source${meta.sources === 1 ? '' : 's'}`));
            main.appendChild(line);
        }

        row.appendChild(main);
        thread.appendChild(row);
        scrollToEnd();
        return row;
    }

    function showExcerpts(filename, results, opener) {
        const excerpts = (results || []).filter((r) => r.filename === filename).map((r) => r.chunk_text || '');
        $('doc-dialog-title').textContent = filename;
        $('doc-dialog-text').textContent = (excerpts.length ? excerpts.join('\n\n---\n\n') : 'No excerpts available for this source.') +
            '\n\nShowing the retrieved excerpts only. Operators can open the full document in the console.';
        openDialog($('doc-dialog'), opener);
    }

    // ---------- sending ----------

    function setBusy(value) {
        busy = value;
        sendBtn.disabled = value;
        sendBtn.setAttribute('aria-busy', String(value));
        $('send-icon').hidden = value;
        $('send-spinner').hidden = !value;
    }

    function failureMessage(status) {
        if (status === 401 || status === 403) return 'Your session has expired. Reload the page and try again.';
        if (status === 429) return 'Too many requests right now. Please wait a moment and try again.';
        if (status === 413) return 'That message or its attachments are too large. Shorten it or remove an image and try again.';
        return `Something went wrong while searching (error ${status}). Your message is still in the box, so you can try again.`;
    }

    async function send() {
        if (busy) return;
        const query = input.value.trim();
        if (!query && attachments.length === 0) return;

        const files = attachments.slice();
        let label = query || '(image attached)';
        if (files.length) label += `\n[${files.length} image${files.length === 1 ? '' : 's'} attached]`;
        addMessage('user', label);
        setBusy(true);
        showTyping();

        try {
            const images = [];
            for (const f of files) {
                images.push({ data_base64: await fileToBase64(f), mime_type: IMAGE_TYPES[extOf(f.name)] });
            }
            const body = { query: query || 'Please analyze the attached content.', top_k: 10, thread_id: getThreadId() };
            if (images.length) body.images = images;

            const res = await fetch('/search', {
                method: 'POST',
                credentials: 'same-origin',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
                signal: AbortSignal.timeout(90_000),
            });
            const data = await res.json().catch(() => ({}));
            removeTyping();
            if (!res.ok) {
                addMessage('bot', failureMessage(res.status), { error: true });
                return;
            }
            addMessage('bot', data.answer || data.message || 'Here are the most relevant passages I found:', {
                results: data.results,
                meta: { totalMs: data.total_ms, sources: (data.sources || []).length },
            });
            if (input.value.trim() === query) input.value = '';
            attachments = [];
            renderAttachments();
        } catch (err) {
            console.warn('Search request failed', err);
            removeTyping();
            addMessage('bot', 'Could not reach the assistant. Check your connection; your message is still in the box, so you can try again.', { error: true });
        } finally {
            setBusy(false);
            input.focus();
        }
    }

    function newConversation() {
        if (busy) { toast('Please wait for the current answer first.', false); return; }
        storage.remove(THREAD_KEY);
        thread.querySelectorAll('.message').forEach((node) => node.remove());
        $('welcome-hero').hidden = false;
        attachments = [];
        renderAttachments();
        input.value = '';
        setSidebarOpen(false);
        history.scrollTop = 0;
        input.focus();
    }

    // ---------- wiring ----------

    document.addEventListener('click', (e) => {
        const target = e.target.closest('[data-action], [data-suggestion]');
        if (!target) return;
        const { action, suggestion } = target.dataset;
        if (action === 'toggle-sidebar') setSidebarOpen(!sidebarIsOpen());
        else if (action === 'new-conversation') newConversation();
        else if (action === 'attach-image') $('chat-file-input').click();
        else if (suggestion) {
            if (busy) return;
            input.value = suggestion;
            send();
        }
    });

    input.addEventListener('keydown', (e) => {
        // Enter sends; Shift+Enter inserts a newline; never send while an IME is composing.
        if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
            e.preventDefault();
            send();
        }
    });

    // Fallback for browsers without CSS `field-sizing`.
    if (!(window.CSS && CSS.supports && CSS.supports('field-sizing', 'content'))) {
        input.addEventListener('input', () => {
            input.style.height = 'auto';
            input.style.height = Math.min(input.scrollHeight, 200) + 'px';
        });
    }

    $('chat-file-input').addEventListener('change', (e) => {
        addAttachments(Array.from(e.target.files || []));
        e.target.value = '';
    });
    sendBtn.addEventListener('click', send);
    backdrop.addEventListener('click', () => setSidebarOpen(false, { returnFocus: true }));

    document.addEventListener('keydown', (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
            e.preventDefault();
            newConversation();
        } else if (e.key === 'Escape' && sidebarIsOpen()) {
            setSidebarOpen(false, { returnFocus: true });
        }
    });

    mobile.addEventListener('change', () => setSidebarOpen(false));

    // ---------- init ----------

    $('new-chat-kbd').textContent = isApple ? '⌘K' : 'Ctrl K';
    setSidebarOpen(false);
    checkHealth();
    setInterval(() => { if (!document.hidden) checkHealth(); }, HEALTH_POLL_MS);
    document.addEventListener('visibilitychange', () => { if (!document.hidden) checkHealth(); });
})();
