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

        # Search configuration
        self.TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', '5'))
        # Similarity thresholds (0.0-1.0)
        self.SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.3'))
        self.DOMAIN_GUARD_THRESHOLD = float(os.getenv('DOMAIN_GUARD_THRESHOLD', '0.5'))
        # Qdrant search parameters
        self.QDRANT_SCORE_THRESHOLD = float(os.getenv('QDRANT_SCORE_THRESHOLD', '0.15'))
        self.MAX_SEARCH_LIMIT = int(os.getenv('MAX_SEARCH_LIMIT', '15'))

        # AI model defaults
        self.MAX_TOKENS = int(os.getenv('MAX_TOKENS', '800'))
        self.TEMPERATURE = float(os.getenv('TEMPERATURE', '0.1'))

        # Demo matching thresholds
        self.DEMO_MATCH_RATIO = float(os.getenv('DEMO_MATCH_RATIO', '0.6'))
        self.DEMO_MIN_MATCH_SCORE = float(os.getenv('DEMO_MIN_MATCH_SCORE', '0.0'))

        # Garbage collection tuning
        self.GC_DEFAULT_INTERVAL = int(os.getenv('GC_DEFAULT_INTERVAL', '1800'))
        self.GC_MIN_INTERVAL = int(os.getenv('GC_MIN_INTERVAL', '1800'))
        self.GC_MAX_INTERVAL = int(os.getenv('GC_MAX_INTERVAL', '3600'))
        # Memory thresholds (MB)
        self.MEMORY_WARNING_THRESHOLD = int(os.getenv('MEMORY_WARNING_THRESHOLD', '1024'))
        self.MEMORY_CRITICAL_THRESHOLD = int(os.getenv('MEMORY_CRITICAL_THRESHOLD', '2048'))
        # Performance settings
        self.GC_MAX_MEMORY_SAMPLES = int(os.getenv('GC_MAX_MEMORY_SAMPLES', '100'))
        self.GC_EMERGENCY_THRESHOLD = float(os.getenv('GC_EMERGENCY_THRESHOLD', '0.90'))
        self.MEMORY_CACHE_DURATION = int(os.getenv('MEMORY_CACHE_DURATION', '3'))
        self.INACTIVITY_RESET_HOURS = int(os.getenv('INACTIVITY_RESET_HOURS', '2'))

        # Logging control: when false, logs will only be emitted to stderr/console
        self.LOG_TO_FILE = os.getenv('LOG_TO_FILE', 'false').lower() in ('1', 'true', 'yes')

settings = Settings()
