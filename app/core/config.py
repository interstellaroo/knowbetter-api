from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database Configuration
    database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/knowbetter"
    postgres_db: str = "knowbetter"
    postgres_user: str = "postgres"
    postgres_password: str = "password"
    
    # Google Search API
    google_search_api_key: str
    google_search_engine_id: str

    class Config:
        env_file = ".env"


settings = Settings()