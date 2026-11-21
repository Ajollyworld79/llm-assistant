// Shared helpers for the chat page and the console. Exposes window.Helios.
(() => {
    'use strict';

    // One breakpoint for layout logic; keep in sync with the CSS `max-width: 900px` queries.
    const mobile = window.matchMedia('(max-width: 900px)');

    // ---- storage that never throws (private mode, blocked cookies) ----
    const memory = new Map();
    const storage = {
        get(key) {
            try { return window.localStorage.getItem(key); } catch (_) { return memory.has(key) ? memory.get(key) : null; }
        },
        set(key, value) {
            memory.set(key, String(value));
            try { window.localStorage.setItem(key, String(value)); } catch (_) { /* memory copy is enough */ }
        },
        remove(key) {
            memory.delete(key);
            try { window.localStorage.removeItem(key); } catch (_) { /* nothing stored */ }
        },
    };

    function escapeHtml(value) {
        return String(value == null ? '' : value)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }

    // ---- toasts: one polite live region, moved into the open dialog so they stay visible ----
    let region = null;

    function toastRegion() {
        if (!region) {
            region = document.createElement('div');
            region.className = 'toast-region';
            region.setAttribute('role', 'status');
            region.setAttribute('aria-live', 'polite');
        }
        const host = document.querySelector('dialog[open]') || document.body;
        if (region.parentElement !== host) host.appendChild(region);
        return region;
    }

    function toast(message, success = true, duration = 4000) {
        const host = toastRegion();
        while (host.children.length >= 3) host.firstElementChild.remove();
        const el = document.createElement('div');
        el.className = `toast ${success ? 'toast-success' : 'toast-error'}`;
        el.textContent = message;
        host.appendChild(el);
        setTimeout(() => {
            el.classList.add('toast-hide');
            setTimeout(() => el.remove(), 300);
        }, success ? duration : duration + 2000);
    }

    // ---- dialogs: native <dialog>, Escape and focus return come from the platform ----
    function openDialog(dialog, opener) {
        if (dialog.open) return;
        dialog._opener = opener || document.activeElement;
        dialog.showModal();
    }

    function closeDialog(dialog, returnValue) {
        if (dialog.open) dialog.close(returnValue);
    }

    function initDialogs(root = document) {
        root.querySelectorAll('dialog.dialog').forEach((dialog) => {
            if (dialog._wired) return;
            dialog._wired = true;
            // A click on the dialog element itself (not its panel content) is a backdrop click.
            dialog.addEventListener('click', (e) => {
                if (e.target === dialog) dialog.close();
            });
            dialog.addEventListener('close', () => {
                const opener = dialog._opener;
                dialog._opener = null;
                if (opener && opener.isConnected && typeof opener.focus === 'function') opener.focus();
            });
        });
        root.querySelectorAll('[data-close-dialog]').forEach((btn) => {
            if (btn._wired) return;
            btn._wired = true;
            btn.addEventListener('click', () => btn.closest('dialog').close());
        });
    }

    const isApple = /Mac|iPhone|iPad/i.test(navigator.platform || navigator.userAgent || '');

    window.Helios = { mobile, storage, escapeHtml, toast, openDialog, closeDialog, initDialogs, isApple };

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => initDialogs());
    else initDialogs();
})();
