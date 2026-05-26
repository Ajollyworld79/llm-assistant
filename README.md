# LLM Assistant

An AI-powered IT-support agent using Qdrant for vector search and Azure OpenAI for generative responses. The system supports document uploads (PDF, TXT, DOCX, CSV), intelligent chunking, embeddings, and Retrieval-Augmented Generation (RAG) for accurate answers based on your documents.

## GDPR setup checklist (production deploy)

Before exposing this demo to customers, complete these steps so the GDPR story holds up:

1. **Azure region in EU.** Deploy the Azure OpenAI resource in `swedencentral` or `westeurope`. Set `AZURE_REGION` env var to match.
2. **Azure abuse-monitoring opt-out.** By default, Microsoft retains prompts for 30 days for abuse review. Submit the modified-data-handling form to disable that retention:
   - <https://aka.ms/oai/additionalusecase>
   - Once approved by Microsoft, set `AZURE_ABUSE_MONITORING_OPT_OUT=true` so `/gdpr/policy` reflects it.
3. **Payload encryption.** Generate a 32-byte key and set `PAYLOAD_ENCRYPTION_KEY`. The Qdrant payload `text` field is then AES-256-GCM encrypted at rest:
   ```bash
   python -c "import os, base64; print(base64.b64encode(os.urandom(32)).decode())"
   ```
4. **Retention.** Defaults are conservative: documents auto-deleted after 7 days (`DOCUMENT_TTL_DAYS`), logs after 30 days (`LOG_RETENTION_DAYS`). Adjust if your customers require shorter retention.
5. **Rate limits.** `RATE_LIMIT_PER_MIN=10`, `RATE_LIMIT_PER_DAY=100`, `DAILY_TOKEN_BUDGET=50000` keep Azure costs bounded on a public demo. Tune for your traffic.
6. **DPA (Data Processing Agreement).** Microsoft's standard Azure DPA covers Azure OpenAI. Link to it from the customer-facing materials. Railway's DPA is at <https://railway.com/legal/dpa>.
7. **Verify EU residency in the data-flow diagram.** Open `/` and click the EU banner — the flow shows browser → Railway EU → Qdrant local → Azure OpenAI (Sweden Central).

## Features

- **Document Upload**: Upload and parse files (PDF, TXT, DOCX, CSV)
- **Intelligent Chunking**: Automatic text splitting into meaningful chunks
- **Embeddings**: Supports SentenceTransformers (local) and Azure OpenAI embeddings
- **Vector Search**: Cosine similarity search in Qdrant database
- **RAG Chat**: Generative responses based on relevant document chunks
- **Local/Persistent Storage**: Qdrant local database for data persistence
- **Demo Mode**: Simulated functionality without external APIs
- **Admin Functions**: Delete documents, reset database

## Architecture

### Backend Modules

- **`main.py`**: Main application with Quart routes for upload, search, and chat
- **`config.py`**: Configuration settings from environment variables
- **`embeddings.py`**: Embedding providers (SentenceTransformers, Azure OpenAI, FastEmbed)
- **`lifecycle.py`**: Application lifecycle management (startup/shutdown, Qdrant initialization)
- **`apimonitor.py`**: API monitoring and metrics
- **`middleware.py`**: Quart middleware for logging and error handling

### How It Works

1. **Upload**: File uploaded → parsed → chunked → embedded → stored in Qdrant
2. **Search**: Query embedded → vector search in Qdrant → returns relevant chunks
3. **Chat**: Search results used as context → Azure OpenAI generates response

## Installation

### Requirements

- Python 3.8+
- pip
- Azure OpenAI account (optional for demo mode)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd llm-assistant

# Install dependencies
pip install -r requirements.txt
cd backend
pip install -r requirements.txt
```

## Setup

### Environment Variables

Create `.env` file in `backend/` folder:

```env
# Demo mode (true/false)
DEMO=false

# Qdrant configuration
QDRANT_URL=  # Leave empty for local
QDRANT_PATH=local_qdrant_db
QDRANT_COLLECTION=documents
QDRANT_FORCE_RECREATE=false

# Embedding configuration
EMBEDDING_PROVIDER=fastembed
EMBEDDING_DIM=384
EMBEDDING_TIMEOUT=10

# Gemini (free Google AI Studio key — required if DEMO=false)
GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_MODEL_FALLBACK=gemini-2.5-flash-lite

# Admin token for protected endpoints
ADMIN_TOKEN=your-secure-token
```

### Local Qdrant

The system automatically uses local Qdrant database in the `local_qdrant_db/` folder. Data persists between restarts.

### Gemini setup

1. Grab a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Set `GEMINI_API_KEY` in your environment (or on Railway as a project variable).
3. Default primary model is `gemini-2.5-flash`, fallback is `gemini-2.0-flash` — both on the free tier.

## Usage

### Start the Server

```bash
cd backend
python app/main.py
```

Server runs on http://127.0.0.1:8002

### API Endpoints

#### Upload Document
```http
POST /upload
Authorization: Bearer <admin-token>
Content-Type: multipart/form-data

file: <your-file>
```

#### List Documents
```http
GET /documents
```

#### Search
```http
POST /search
Content-Type: application/json

{
  "query": "your search",
  "top_k": 5
}
```

Returns:
```json
{
  "results": [...],
  "sources": ["file1.pdf", "file2.txt"],
  "answer": "Generative answer...",
  "demo": false
}
```

#### Delete Document
```http
DELETE /document/<filename>
Authorization: Bearer <admin-token>
```

#### Reset Database
```http
POST /reset
Authorization: Bearer <admin-token>
```

### Frontend

Open `templates/index.html` in browser or use a web server.

## Configuration Details

### Embedding Providers

- **sentence**: SentenceTransformers (local, 384 dim, all-MiniLM-L6-v2)
- **azure**: Azure OpenAI embeddings (cloud, 3072 dim for large)
- **fastembed**: FastEmbed (local, various models)

### Chunking

- Chunk size: 200 words
- Overlap: 50 words
- Based on sentences for natural breaks

### Qdrant

- Distance: Cosine similarity
- Collection name: `documents`
- Persistent storage in local folder

## Development

### Project Structure

```
llm-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py          # Main app
│   │   ├── config.py        # Config
│   │   ├── embeddings.py    # Embedding logic
│   │   ├── lifecycle.py     # Startup/shutdown
│   │   ├── apimonitor.py    # Monitoring
│   │   ├── middleware.py    # Middleware
│   │   └── tests/           # Tests
│   ├── static/              # CSS/JS
│   ├── templates/           # HTML templates
│   └── .env                 # Environment variables
├── local_qdrant_db/         # Qdrant data
└── requirements.txt
```

### Debugging

- Logs: Check terminal output
- Demo mode: Set `DEMO=true` for simulation
- Admin endpoints: Use `ADMIN_TOKEN` in Authorization header

## Security

- Admin endpoints protected with Bearer token
- No direct file access
- Environment variables for sensitive data
- CORS configuration for frontend

## Troubleshooting

### Common Issues

1. **Embedding dimension mismatch**: Delete `local_qdrant_db/` and restart
2. **Azure API error**: Check credentials and endpoints
3. **Upload fails**: Check file type support and admin token
4. **No results**: Check if files are uploaded and embedded

### Logs

Check Quart logs for detailed error info.

## Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push and create PR

## License

MIT License

Copyright (c) 2025 Gustav Christensen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.