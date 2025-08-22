# app/core/config.py
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App metadata
    app_name: str = Field("Email Triage API", alias="APP_NAME")
    app_version: str = Field("0.1.0", alias="APP_VERSION")

    # Common envs
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    database_url: str | None = Field(default=None, alias="DATABASE_URL")

    # pydantic-settings config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="ignore",   # ignore unknown env vars (prevents crashes)
    )

settings = Settings()

