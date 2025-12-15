import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    def __init__(self):
        self.demo = os.getenv('DEMO', 'true').lower() in ('1', 'true', 'yes')
        self.qdrant_url = os.getenv('QDRANT_URL') or None
        self.qdrant_api_key = os.getenv('QDRANT_API_KEY') or None
        self.qdrant_collection = os.getenv('QDRANT_COLLECTION', 'documents')
        self.embedding_provider = os.getenv('EMBEDDING_PROVIDER', 'fastembed')
        self.embedding_dim = int(os.getenv('EMBEDDING_DIM', '64'))
        # Admin token for protecting upload/reset endpoints
        self.admin_token = os.getenv('ADMIN_TOKEN') or None

settings = Settings()
