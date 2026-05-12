# src/models/model_registry.py
import joblib
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

import torch

from src.data.preprocessing import DataPreprocessor
from src.models.factory import TaskModelFactory


class ModelRegistry:
    _instance = None
    _initialized = False

    def __init__(self, models_directory: str = "models"):
        if not ModelRegistry._initialized:
            self.models_directory = Path(models_directory)
            self.models_directory.mkdir(exist_ok=True)
            self._models = {}
            self._model_metadata = {}
            self.logger = logging.getLogger(__name__)
            ModelRegistry._initialized = True

    @classmethod
    def get_instance(cls, models_directory: str = "models") -> 'ModelRegistry':
        """
        Singleton pattern to ensure single instance
        """
        if cls._instance is None:
            cls._instance = cls(models_directory)
        return cls._instance

    def register_model(self, name: str, model: Any, metadata: Optional[Dict] = None) -> bool:
        """
        Register a model in the registry
        """
        try:
            self._models[name] = model
            self._model_metadata[name] = metadata or {}

            model_path = self.models_directory / f"{name}.joblib"
            joblib.dump(model, model_path)

            metadata_path = self.models_directory / f"{name}_metadata.joblib"
            joblib.dump(self._model_metadata[name], metadata_path)

            self.logger.info(f"Model '{name}' registered successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to register model '{name}': {str(e)}")
            return False

    def get_model(self, name: str) -> Optional[Any]:

        try:
            # If model is in memory, return it
            if name in self._models:
                return self._models[name]

            # Try to load from disk
            model_path = self.models_directory / f"{name}.joblib"
            if model_path.exists():
                model = joblib.load(model_path)
                self._models[name] = model

                # Load metadata if exists
                metadata_path = self.models_directory / f"{name}_metadata.joblib"
                if metadata_path.exists():
                    self._model_metadata[name] = joblib.load(metadata_path)

                self.logger.info(f"Model '{name}' loaded from disk")
                return model

            self.logger.warning(f"Model '{name}' not found")
            return None

        except Exception as e:
            self.logger.error(f"Failed to get model '{name}': {str(e)}")
            return None

    def list_models(self) -> List[str]:
        """
        List all available models
        """
        try:
            # Get models from memory
            memory_models = set(self._models.keys())

            # Get models from disk
            disk_models = set()
            if self.models_directory.exists():
                for file_path in self.models_directory.glob("*.joblib"):
                    if not file_path.name.endswith("_metadata.joblib"):
                        model_name = file_path.stem
                        disk_models.add(model_name)

            return list(memory_models | disk_models)

        except Exception as e:
            self.logger.error(f"Failed to list models: {str(e)}")
            return []

    def list_loaded_models(self) ->List[str]:
        return list(self._models.keys())

    def get_model_metadata(self, name: str) -> Dict[str, Any]:
        """
        Get metadata for a specific model
        """
        try:
            if name in self._model_metadata:
                return self._model_metadata[name]

            # Try to load metadata from disk
            metadata_path = self.models_directory / f"{name}_metadata.joblib"
            if metadata_path.exists():
                metadata = joblib.load(metadata_path)
                self._model_metadata[name] = metadata
                return metadata

            return {}

        except Exception as e:
            self.logger.error(f"Failed to get metadata for model '{name}': {str(e)}")
            return {}

    def clear_all_models(self) -> bool:
        try:
            num_cleared = len(self._models)
            self._models.clear()
            self._model_metadata.clear()
            self.logger.info(f"Registry cleared successfully. Removed {num_cleared} models from memory.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear the model registry: {str(e)}")
            return False


    def remove_model(self, name: str) -> bool:
        try:
            # Remove from memory
            if name in self._models:
                del self._models[name]
            if name in self._model_metadata:
                del self._model_metadata[name]

            # Remove from disk
            model_path = self.models_directory / f"{name}.joblib"
            if model_path.exists():
                model_path.unlink()

            metadata_path = self.models_directory / f"{name}_metadata.joblib"
            if metadata_path.exists():
                metadata_path.unlink()

            self.logger.info(f"Model '{name}' removed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to remove model '{name}': {str(e)}")
            return False

    def get_registry_info(self) -> Dict[str, Any]:
        """
        Get information about the registry
        """
        return {
            'models_directory': str(self.models_directory),
            'loaded_models': list(self._models.keys()),
            'available_models': self.list_models(),
            'total_models': len(self.list_models())
        }

    def load_model(self, model_type: str, model_path: str) -> Dict[str, Any]:
        """
        Orchestrează încărcarea unui pachet complet de model PyTorch.
        """
        try:
            # 1. Încarcă fișierul .pth
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)

            # 2. Recreează arhitectura modelului
            if 'priority' in model_type:
                model = TaskModelFactory.create_classifier(**checkpoint['model_config'])
                preprocessor_path = '../../artifacts/classification_preprocessors.pkl'
            elif 'duration' in model_type:
                model = TaskModelFactory.create_regressor(**checkpoint['model_config'])
                preprocessor_path = '../../artifacts/regression_preprocessors.pkl'
            else:
                raise ValueError(f"Tip de model necunoscut: {model_type}")

            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            trainer = TaskModelFactory.create_trainer(model)
            preprocessor = DataPreprocessor.load(preprocessor_path)

            model_package = {
                'model': model,
                'trainer': trainer,
                'preprocessor': preprocessor,
                'model_config': checkpoint['model_config']
            }

            self._models[model_type] = model_package  # Stocăm în memorie pentru acces ulterior
            return model_package

        except Exception as e:
            self.logger.error(f"A eșuat încărcarea pachetului de model pentru '{model_type}': {e}")
            raise