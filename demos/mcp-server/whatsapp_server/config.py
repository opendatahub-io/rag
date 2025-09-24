"""Configuration management for WhatsApp server."""

import os
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Evolution API Configuration
    evolution_api_url: str = Field(..., env="EVOLUTION_API_URL")
    evolution_api_id: str = Field(..., env="EVOLUTION_API_ID") 
    evolution_api_token: str = Field(..., env="EVOLUTION_API_TOKEN")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # WhatsApp Configuration
    redbank_group_name: str = Field(default="RedBank", env="REDBANK_GROUP_NAME")
    welcome_message: str = Field(default="Welcome to RedBank", env="WELCOME_MESSAGE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
