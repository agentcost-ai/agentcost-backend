"""
AgentCost Backend Configuration

Loads settings from environment variables.
"""

import os
import secrets
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List
from functools import lru_cache


def get_version() -> str:
    # Look for VERSION file in project root
    possible_paths = [
        Path(__file__).parent.parent.parent.parent / "VERSION",  # From app/
        Path(__file__).parent.parent.parent / "VERSION",
        Path("VERSION"),
        Path("../VERSION"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return path.read_text().strip()
    
    return "0.1.0"  # Default fallback


def generate_dev_secret() -> str:
    """Generate a random secret for development only"""
    return secrets.token_urlsafe(32)


class Settings(BaseSettings):
    """Application configuration settings"""   
    app_name: str = "AgentCost API"
    app_version: str = get_version()  
    debug: bool = False
    environment: str = "development"  # development, staging, production
    
    # Database
    # Default: SQLite for development
    database_url: str = "sqlite+aiosqlite:///./agentcost.db"
    
    # Authentication - set via environment in production!
    secret_key: str = ""
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 7
    
    # API Keys
    api_key_prefix: str = "sk_"
    
    # CORS - Add your frontend domains in production
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    
    # Rate limiting
    rate_limit_requests: int = 100  # requests per minute
    rate_limit_period: int = 60  # seconds
    
    # Batch processing
    max_batch_size: int = 100  # max events per batch request
    
    # Pricing sync settings
    # Auto-sync pricing from LiteLLM on startup
    # Set to False in production - use POST /v1/pricing/sync/litellm from Admin Dashboard instead
    auto_sync_pricing_on_startup: bool = False
    # Sync interval in hours (0 = no background sync, just startup)
    pricing_sync_interval_hours: int = 24
    
    # Email - for verification and password reset emails
    resend_api_key: str = ""
    resend_sender_email: str = "noreply@agentcost.dev"
    resend_sender_name: str = "AgentCost"
    frontend_url: str = "http://localhost:3000"
    
    # Legal Policy Versions
    # IMPORTANT: Increment these when policies are updated
    # Users must re-accept updated policies before using the app
    terms_of_service_version: str = "1.0"
    privacy_policy_version: str = "1.0"
    
    # Server configuration (for uvicorn when run directly)
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Extended session duration (for "remember me" feature)
    extended_session_days: int = 30
    
    # External pricing sources (with fallback defaults)
    litellm_pricing_url: str = (
        "https://raw.githubusercontent.com/BerriAI/litellm/main/"
        "model_prices_and_context_window.json"
    )
    openrouter_models_url: str = "https://openrouter.ai/api/v1/models"
    
    # Rate limiting backend: "memory" (default) or "redis" (for multi-instance)
    rate_limit_backend: str = "memory"
    redis_url: str = ""
    
    # Request size limit (MB) - protects against oversized payloads
    max_request_size_mb: int = 10
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )
    
    def model_post_init(self, __context) -> None:
        """Validate settings after initialization"""
        # Generate dev secret if not provided
        if not self.secret_key:
            if self.environment == "production":
                raise ValueError(
                    "SECRET_KEY environment variable is required in production! "
                    "Generate one with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
                )
            # Use random secret for development (regenerates on restart)
            object.__setattr__(self, 'secret_key', generate_dev_secret())
        
        # Warn about SQLite in production
        if self.environment == "production" and "sqlite" in self.database_url:
            import warnings
            warnings.warn(
                "Using SQLite in production is not recommended! "
                "Set DATABASE_URL to a PostgreSQL connection string."
            )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


def clear_settings_cache():
    """Clear cached settings (useful for testing)"""
    get_settings.cache_clear()
