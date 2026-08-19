"""Environment-backed configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    google_cloud_project: str
    google_cloud_location: str = "us-central1"
    vertex_model: str = "gemini-2.5-flash"
    local_user_id: str = "local-user"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def sqlalchemy_url(self) -> str:
        return self.database_url.replace("postgresql://", "postgresql+psycopg://", 1)


def load_settings() -> Settings:
    """Read required settings from the environment at application startup."""
    return Settings()  # pyright: ignore[reportCallIssue]  # Pydantic reads these from env.
