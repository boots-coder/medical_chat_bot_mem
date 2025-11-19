"""
项目配置管理
使用 pydantic-settings 从环境变量加载配置
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    """应用配置"""

    # API Configuration
    api_provider: str = "ark"
    api_key: str
    api_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    api_model: str = "deepseek-v3-250324"

    # Database Configuration
    sqlite_db_path: str = "./data/sessions.db"
    chroma_persist_dir: str = "./data/chroma"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str

    # JWT Configuration
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    token_expire_minutes: int = 30

    # Session Configuration
    session_timeout_minutes: int = 30
    max_dialogue_turns: int = 100

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # CORS Configuration
    allowed_origins: str = "http://localhost:3000,http://localhost:8000"

    # Clustering Configuration
    cluster_min_size: int = 3
    cluster_min_samples: int = 2
    sbert_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # Memory Configuration
    short_term_max_tokens: int = 2000
    short_term_max_turns: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    @property
    def allowed_origins_list(self) -> List[str]:
        """将逗号分隔的origins转换为列表"""
        return [origin.strip() for origin in self.allowed_origins.split(",")]


# 全局配置实例
settings = Settings()


# 用于测试的配置验证
if __name__ == "__main__":
    print("=== Configuration Loaded ===")
    print(f"API Provider: {settings.api_provider}")
    print(f"API Model: {settings.api_model}")
    print(f"SQLite DB: {settings.sqlite_db_path}")
    print(f"Chroma Dir: {settings.chroma_persist_dir}")
    print(f"Neo4j URI: {settings.neo4j_uri}")
    print(f"Server: {settings.host}:{settings.port}")
    print(f"Debug Mode: {settings.debug}")
    print(f"Allowed Origins: {settings.allowed_origins_list}")
