# Railway Deployment Guide

## Prerequisites
1. Push your code to GitHub
2. Create account at [railway.app](https://railway.app)

## Deployment Steps

### 1. Create New Project
- Go to Railway dashboard
- Click "New Project"
- Select "Deploy from GitHub repo"
- Choose your `llm-assistant` repository

### 2. Add Environment Variables
Go to your service → Variables tab and add:

```
DEMO=false
QDRANT_PATH=/app/qdrant_data
QDRANT_COLLECTION=documents
QDRANT_FORCE_RECREATE=false
EMBEDDING_PROVIDER=azure
EMBEDDING_DIM=3072
EMBEDDING_TIMEOUT=10
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_ENDPOINT=<your-endpoint>
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_DEPLOYMENT_CHAT=<your-chat-deployment>
AZURE_DEPLOYMENT_EMBED=<your-embed-deployment>
ADMIN_TOKEN=<generate-secure-token>
```

### 3. Add Volume for Qdrant Data
- Go to your service → Settings → Volumes
- Click "New Volume"
- Mount path: `/app/qdrant_data`
- Size: 1GB (free tier)

### 4. Deploy
Railway will automatically:
- Detect Python
- Install dependencies from `requirements.txt`
- Run the start command from `railway.json` or `Procfile`
- Assign a public URL

### 5. Update Your CV Site
Once deployed, add the Railway URL to your personal-project links.

## Free Tier Limits
- $5 credit/month (no card required initially)
- 512MB RAM
- 1GB storage
- Should be enough for both projects

## Monitoring
- Check logs in Railway dashboard
- Monitor usage in "Usage" tab
- Set up usage alerts to stay within free tier

## Troubleshooting
- If build fails, check logs in Railway dashboard
- Ensure all environment variables are set
- Volume must be mounted before first deploy
