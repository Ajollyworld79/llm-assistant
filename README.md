# LLM Assistant

An AI-powered document assistant using Qdrant for vector search and Azure OpenAI for generative responses. The system supports document uploads (PDF, TXT, DOCX, CSV), intelligent chunking, embeddings, and Retrieval-Augmented Generation (RAG) for accurate answers based on your documents.

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
EMBEDDING_PROVIDER=azure  # or 'sentence' for local
EMBEDDING_DIM=3072  # 384 for sentence, 3072 for azure large
EMBEDDING_TIMEOUT=10

# Azure OpenAI (required if DEMO=false)
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_DEPLOYMENT_CHAT=gpt-4
AZURE_DEPLOYMENT_EMBED=text-embedding-3-large

# Admin token for protected endpoints
ADMIN_TOKEN=your-secure-token
```

### Local Qdrant

The system automatically uses local Qdrant database in the `local_qdrant_db/` folder. Data persists between restarts.

### Azure OpenAI Setup

1. Create Azure OpenAI resource
2. Deploy models:
   - `gpt-4` or `gpt-35-turbo` for chat
   - `text-embedding-3-large` for embeddings (3072 dim)
3. Copy API key, endpoint, and deployment names to `.env`

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