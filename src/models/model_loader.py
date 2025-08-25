import torch
from pathlib import Path
from typing import Tuple
from src.data.preprocessing import DataPreprocessor
from src.models.factory import TaskModelFactory


class ModelLoader:
    """Utility class for loading trained models and preprocessors"""

    def __init__(self , model_dir: str = "./"):
        self.model_dir = Path(model_dir)
        self.classification_model = None
        self.regression_model = None
        self.classification_processor = None
        self.regression_processor = None

    def load_classification_model(self) -> Tuple[object , DataPreprocessor]:
        """Load classification model and its preprocessor"""
        try:
            # Load preprocessor
            processor = DataPreprocessor()
            processor_path = self.model_dir / 'artifacts/classification_preprocessors.pkl'
            processor.load(str(processor_path) , )

            # Load model
            model_path = self.model_dir / 'artifacts/classification_model.pth'
            checkpoint = torch.load(model_path , map_location='cpu')

            model = TaskModelFactory.create_classifier(**checkpoint['model_config'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            trainer = TaskModelFactory.create_trainer(model)

            self.classification_model = trainer
            self.classification_processor = processor

            return trainer , processor

        except Exception as e:
            raise RuntimeError(f"Failed to load classification model: {e}")

    def load_regression_model(self) -> Tuple[object , DataPreprocessor]:
        """Load regression model and its preprocessor"""
        try:
            # Load preprocessor
            processor = DataPreprocessor()
            processor_path = self.model_dir / 'artifacts/regression_preprocessors.pkl'
            processor.load(str(processor_path) , )

            # Load model
            model_path = self.model_dir / 'artifacts/regression_model.pth'
            checkpoint = torch.load(model_path , map_location='cpu')

            model = TaskModelFactory.create_regressor(**checkpoint['model_config'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            trainer = TaskModelFactory.create_trainer(model)

            self.regression_model = trainer
            self.regression_processor = processor

            return trainer , processor

        except Exception as e:
            raise RuntimeError(f"Failed to load regression model: {e}")

    def load_all_models(self) -> bool:
        """Load both models and return success status"""
        try:
            self.load_classification_model()
            self.load_regression_model()
            return True
        except Exception as e:
            print(f"Failed to load models: {e}")
            return False