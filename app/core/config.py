from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Email Triage & Reply API"
    APP_VERSION: str = "1.0.0"
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False

    class Config:
        env_file = ".env"

settings = Settings()
