from dataclasses import dataclass , field
from typing import Dict , List , Optional
import os


@dataclass
class DatabaseConfig:
    host: str = os.getenv('DB_HOST' , 'localhost')
    port: int = int(os.getenv('DB_PORT' , 5432))
    user: str = os.getenv('DB_USER' , 'postgres')
    password: str = os.getenv('DB_PASSWORD' , '101102')
    database: str = os.getenv('DB_NAME' , 'licenta_db')


@dataclass
class ModelConfig:
    classifier_hidden_sizes: Optional[List[int]] = None
    regressor_hidden_sizes: Optional[List[int]] = None
    dropout_rate: float = 0.4
    batch_size: int = 64
    learning_rate: float = 0.001
    max_epochs: int = 200
    early_stopping_patience: int = 50

    def __post_init__(self):
        if self.classifier_hidden_sizes is None:
            self.classifier_hidden_sizes = [256 , 128]
        if self.regressor_hidden_sizes is None:
            self.regressor_hidden_sizes = [512 , 256 , 128]


@dataclass
class AppConfig:
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    artifacts_path: str = "artifacts/"
    confidence_threshold: float = 0.7