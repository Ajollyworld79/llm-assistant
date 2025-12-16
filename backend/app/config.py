# LLM Assistant - Configuration
# Created by Gustav Christensen
# Date: December 2025
# Description: Configuration settings and environment variable loading

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    def __init__(self):
        self.demo = os.getenv('DEMO', 'false').lower() in ('1', 'true', 'yes')
        self.qdrant_url = os.getenv('QDRANT_URL') or None
        self.qdrant_path = os.getenv('QDRANT_PATH') or os.path.join(os.path.dirname(__file__), '..', '..', 'local_qdrant_db')
        self.qdrant_api_key = os.getenv('QDRANT_API_KEY') or None
        self.qdrant_collection = os.getenv('QDRANT_COLLECTION', 'documents')
        # Whether to force recreate the collection on startup (defaults to false to avoid deleting data)
        self.qdrant_force_recreate = os.getenv('QDRANT_FORCE_RECREATE', 'false').lower() in ('1', 'true', 'yes')
        # Default to sentence-transformers for higher semantic quality; fallback to fastembed if unavailable
        self.embedding_provider = os.getenv('EMBEDDING_PROVIDER', 'sentence')
        self.embedding_dim = int(os.getenv('EMBEDDING_DIM', '384'))
        # Maximum seconds to wait for embedding provider operations (init/encode)
        self.embedding_timeout = int(os.getenv('EMBEDDING_TIMEOUT', '10'))
        
        # Azure OpenAI
        self.azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        self.azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.azure_openai_api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
        self.azure_deployment_chat = os.getenv('AZURE_DEPLOYMENT_CHAT')
        self.azure_deployment_embed = os.getenv('AZURE_DEPLOYMENT_EMBED')

        # Admin token for protecting upload/reset endpoints
        self.admin_token = os.getenv('ADMIN_TOKEN') or None

settings = Settings()
