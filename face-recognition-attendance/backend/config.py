from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db: str = "attendance_db"
    face_model: str = "facenet"
    custom_model_path: str = ""


settings = Settings()