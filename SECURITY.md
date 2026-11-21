# Security Policy

## Reporting a vulnerability

Please report vulnerabilities privately by email to **guch79@gmail.com**. Do not open a public issue for security problems. Include a description, reproduction steps and the affected endpoint or file. You will get an acknowledgement as soon as possible, and a fix or mitigation plan after triage.

## Supported versions

Only the `main` branch is supported. Fixes are applied to `main` and deployed from there.

## Security model

- **Fail-closed admin auth.** Every admin route (documents, conversations, upload, `/admin/*`, deletes) requires `ADMIN_TOKEN`. If the token is not configured, those routes refuse all requests.
- **Operator session cookie.** The console logs in via `POST /backend/api/session` and receives an HttpOnly, Secure, SameSite=Strict, HMAC-signed cookie valid for 12 hours. Cookie-authenticated requests must also send `X-Requested-With: helios` as a CSRF guard. Scripts may use `Authorization: Bearer <ADMIN_TOKEN>` instead.
- **Search access.** `POST /search` requires a Bearer `SEARCH_API_TOKEN`/`ADMIN_TOKEN` or the signed chat-session cookie.
- **Encrypted payloads at rest.** Chunk text is stored in Qdrant as AES-256-GCM ciphertext (`enc:v1:` prefix) when `PAYLOAD_ENCRYPTION_KEY` is set.
- **Rate limits and token budget.** Per-IP, per-session and global daily token limits bound abuse and spend.
- **Strict CSP.** `default-src 'self'` with no inline scripts or styles and no third-party origins; all browser libraries are served from `/static/vendor/`.
- **No secrets in the repository.** Configuration comes from environment variables; `.env.example` contains placeholders only.
