import logging
from typing import Dict , Any , Optional , List , Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

from src.models.model_registry import ModelRegistry
from src.data.preprocessing import DataPreprocessor
from src.core.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for model predictions"""
    priority_prediction: Optional[str] = None
    priority_confidence: Optional[float] = None
    duration_prediction: Optional[float] = None
    duration_confidence: Optional[float] = None
    completion_probability: Optional[float] = None
    difficulty_score: Optional[float] = None
    metadata: Optional[Dict[str , Any]] = None
    prediction_timestamp: datetime = None

    def __post_init__(self):
        if self.prediction_timestamp is None:
            self.prediction_timestamp = datetime.now()


class ModelService:
    """
    Service responsible for coordinating ML model operations.

    This service provides a high-level interface for making predictions
    while handling model loading, preprocessing, and error management.
    """

    def __init__(self , model_registry: ModelRegistry , config: AppConfig = None):
        self.model_registry = model_registry
        self.config = config or AppConfig()
        self._model_status = { }
        self._preprocessors = { }

        # Initialize models on service creation
        self._initialize_models()

    def _initialize_models(self):
        """Initialize and load all required models"""
        try:
            # Define model paths
            model_paths = {
                'priority_classifier':'../../artifacts/classification_model.pth' ,
                'duration_regressor':'../../artifacts/regression_model.pth'
            }

            # Load each model
            for model_type , model_path in model_paths.items():
                try:
                    self._load_model_with_preprocessor(model_type , model_path)
                except Exception as e:
                    logger.warning(f"Failed to load {model_type}: {str(e)}")
                    self._model_status[model_type] = 'unavailable'

            available_models = [k for k , v in self._model_status.items() if v == 'available']
            logger.info(f"ModelService initialized with {len(available_models)} available models: {available_models}")

        except Exception as e:
            logger.error(f"Failed to initialize ModelService: {str(e)}")

    def _load_model_with_preprocessor(self , model_type: str , model_path: str):
        """Load model and its associated preprocessor"""
        try:
            # Load model through registry
            model_package = self.model_registry.load_model(model_type , model_path)

            # Store preprocessor if available
            if 'preprocessor' in model_package and model_package['preprocessor']:
                self._preprocessors[model_type] = model_package['preprocessor']
            else:
                # Create new preprocessor and load from file
                preprocessor = DataPreprocessor()
                preprocessor_path = self._get_preprocessor_path(model_type)

                try:
                    preprocessor.load(preprocessor_path , )
                    self._preprocessors[model_type] = preprocessor
                except Exception as e:
                    logger.warning(f"Could not load preprocessor for {model_type}: {str(e)}")
                    self._preprocessors[model_type] = None

            self._model_status[model_type] = 'available'
            logger.info(f"Successfully loaded {model_type}")

        except Exception as e:
            logger.error(f"Failed to load {model_type}: {str(e)}")
            self._model_status[model_type] = 'unavailable'

    def _get_preprocessor_path(self , model_type: str) -> str:
        """Get preprocessor path for model type"""
        if 'classification' in model_type or 'priority' in model_type:
            return '../../artifacts/classification_preprocessors.pkl'
        elif 'regression' in model_type or 'duration' in model_type:
            return '../../artifacts/regression_preprocessors.pkl'
        else:
            return f'../../artifacts/{model_type}_preprocessors.pkl'

    def predict(self , task_data: Dict[str , Any]) -> PredictionResult:
        """
        Generate predictions for a task using available ML models.
        Args:
            task_data: Dictionary containing task information
        Returns:
            PredictionResult with all available predictions
        Raises:
            InvalidTaskDataError: If task data is invalid
            PredictionError: If prediction fails unexpectedly
        """
        try:
            # Validate input data
            validated_data = self._validate_task_data(task_data)

            # Initialize result container
            result = PredictionResult()

            # Make priority prediction
            if self._model_status.get('priority_classifier') == 'available':
                try:
                    result.priority_prediction , result.priority_confidence = self._predict_priority(validated_data)
                except Exception as e:
                    logger.warning(f"Priority prediction failed: {str(e)}")

            # Make duration prediction
            if self._model_status.get('duration_regressor') == 'available':
                try:
                    result.duration_prediction , result.duration_confidence = self._predict_duration(validated_data)
                except Exception as e:
                    logger.warning(f"Duration prediction failed: {str(e)}")

            # Calculate difficulty score based on available predictions
            result.difficulty_score = self._calculate_difficulty_score(result)

            # Add metadata
            result.metadata = {
                'available_models':[k for k , v in self._model_status.items() if v == 'available'] ,
                'task_features_used':len(validated_data) ,
                'service_version':'1.0.0'
            }

            logger.debug(f"Generated predictions for task: {validated_data.get('task_name' , 'Unknown')}")
            return result

        except InvalidTaskDataError:
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise PredictionError(f"Failed to generate predictions: {str(e)}")

    def _validate_task_data(self , task_data: Dict[str , Any]) -> Dict[str , Any]:
        """Validate and clean task data"""

        required_fields = ['name' , 'description']
        for field in required_fields:
            if field not in task_data or not task_data[field]:
                raise InvalidTaskDataError(f"Missing or empty required field: {field}")

        # Convert and validate dates
        validated = task_data.copy()
        for date_field in ['creation_date' , 'due_date']:
            if date_field in validated and isinstance(validated[date_field] , str):
                try:
                    validated[date_field] = pd.to_datetime(validated[date_field])
                except:
                    logger.warning(f"Could not parse date field {date_field}")
                    validated[date_field] = datetime.now()

        return validated

    def _predict_priority(self , task_data: Dict[str , Any]) -> Tuple[str , float]:
        """Predict task priority using classification model"""
        try:
            # Get model components
            model_package = self.model_registry.get_model('priority_classifier')
            trainer = model_package['trainer']
            preprocessor = self._preprocessors.get('priority_classifier')

            if not preprocessor:
                raise PredictionError("No preprocessor available for priority classification")

            # Prepare features
            features = self._prepare_features_for_prediction(task_data , preprocessor , 'classification')

            # Make prediction
            prediction , _ , confidence_scores = trainer.predict(features.reshape(1 , -1) , return_confidence=True)

            # Map prediction to label
            status_labels = { 0:'Low' , 1:'Medium' , 2:'High' }
            priority_label = status_labels.get(prediction[0] , 'Medium')
            confidence = float(confidence_scores[0]) if confidence_scores is not None else 0.5

            return priority_label , confidence

        except Exception as e:
            raise PredictionError(f"Priority prediction failed: {str(e)}")

    def _predict_duration(self , task_data: Dict[str , Any]) -> Tuple[float , float]:
        """Predict task duration using regression model"""
        try:
            # Get model components
            model_package = self.model_registry.get_model('duration_regressor')
            trainer = model_package['trainer']
            preprocessor = self._preprocessors.get('duration_regressor')

            if not preprocessor:
                raise PredictionError("No preprocessor available for duration regression")

            # Prepare features
            features = self._prepare_features_for_prediction(task_data , preprocessor , 'regression')

            # Make prediction
            prediction = trainer.predict(features.reshape(1 , -1))
            duration = max(0.1 , float(prediction[0]))  # Ensure minimum duration

            # Calculate confidence (simplified approach)
            confidence = 0.75  # Fixed confidence for now

            return duration , confidence

        except Exception as e:
            raise PredictionError(f"Duration prediction failed: {str(e)}")

    def _prepare_features_for_prediction(self , task_data: Dict[str , Any] , preprocessor: DataPreprocessor ,
                                         model_type: str) -> np.ndarray:
        """Prepare features for model prediction"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame([task_data])

            df['creation_date'] = pd.to_datetime(df.get('creation_date', datetime.now()))
            df['due_date'] = pd.to_datetime(df.get('due_date', datetime.now()))

            df['creation_day_of_week'] = df['creation_date'].dt.dayofweek
            df['creation_hour'] = df['creation_date'].dt.hour
            df['creation_month'] = df['creation_date'].dt.month
            df['creation_quarter'] = df['creation_date'].dt.quarter

            df['description_length'] = df['description'].str.len().fillna(0)
            df['task_name_length'] = df['name'].str.len().fillna(0)

            # Add missing columns with defaults
            for col, avg_val in preprocessor.global_averages.items():
                if col not in df.columns:
                    df[col] = avg_val

            # Add categorical defaults
            categorical_cols = ['board_name' , 'client_name' , 'email']
            for col in categorical_cols:
                if col not in df.columns:
                    df[col] = "Unknown"

            df_processed = preprocessor.engineer_features(df, fit_mode=False)

            feature_names = preprocessor.feature_names

            if 'description' in df.columns:
                df['description_length'] = df['description'].str.len()

            if 'name' in df.columns:
                df['task_name_length'] = df['name'].str.len()

            for col in feature_names:
                if col not in df_processed.columns:
                    df_processed[col] = 0

            df_final = df_processed[feature_names].fillna(0)

            scaled_features = preprocessor.scaler.transform(df_final.values)

            return scaled_features

        except KeyError as e:
            logger.error(f"Feature preparation failed due to missing key in DataFrame: {e}")
            raise PredictionError(f"Feature preparation failed due to missing data: {e}")

        except Exception as e:
            logger.error(f"An unexpected error occurred during feature preparation: {e}")
            raise PredictionError(f"An unexpected error occurred during feature preparation: {str(e)}")



    def _calculate_difficulty_score(self , result: PredictionResult) -> float:
        """Calculate overall difficulty score based on available predictions"""
        score = 0.5  # Default neutral score

        if result.duration_prediction is not None:
            duration_factor = min(result.duration_prediction / 8.0 , 1.0)  # Normalize to 8-hour max
            score += duration_factor * 0.4

        # Factor in priority (higher priority might indicate complexity)
        if result.priority_prediction is not None:
            priority_mapping = { 'Low':0.2 , 'Medium':0.5 , 'High':0.8 }
            priority_factor = priority_mapping.get(result.priority_prediction , 0.5)
            score += priority_factor * 0.3

        # Factor in confidence (lower confidence = more uncertainty = potentially more difficult)
        if result.priority_confidence is not None and result.duration_confidence is not None:
            avg_confidence = (result.priority_confidence + result.duration_confidence) / 2
            uncertainty_factor = 1 - avg_confidence
            score += uncertainty_factor * 0.3

        return min(max(score , 0.0) , 1.0)  # Clamp between 0 and 1

    def get_model_health(self) -> Dict[str , Any]:
        """Get health status of all models"""
        return {
            'status':self._model_status.copy() ,
            'total_models':len(self._model_status) ,
            'available_models':len([s for s in self._model_status.values() if s == 'available']) ,
            'loaded_models':self.model_registry.list_loaded_models() ,
            'last_check':datetime.now().isoformat() ,
            'preprocessors_loaded':len([k for k , v in self._preprocessors.items() if v is not None])
        }

    def is_service_available(self) -> bool:
        """Check if at least one model is available for predictions"""
        return any(status == 'available' for status in self._model_status.values())

    def reload_models(self) -> Dict[str , str]:
        """Reload all models (useful for model updates)"""
        logger.info("Reloading all models...")

        # Clear current models
        self.model_registry.clear_all_models()
        self._model_status.clear()
        self._preprocessors.clear()

        # Reinitialize
        self._initialize_models()

        return self._model_status.copy()

    def predict_batch(self , tasks_data: List[Dict[str , Any]]) -> List[PredictionResult]:
        """
        Generate predictions for multiple tasks efficiently.

        Args:
            tasks_data: List of task dictionaries

        Returns:
            List of PredictionResult objects
        """
        results = []

        try:
            logger.info(f"Starting batch prediction for {len(tasks_data)} tasks")

            for i , task_data in enumerate(tasks_data):
                try:
                    result = self.predict(task_data)
                    results.append(result)
                except Exception as e:
                    # Create empty result for failed predictions
                    failed_result = PredictionResult()
                    failed_result.metadata = { 'error':str(e) , 'task_index':i }
                    results.append(failed_result)
                    logger.warning(f"Failed to predict task {i}: {str(e)}")

            logger.info(f"Batch prediction completed: {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise PredictionError(f"Failed to generate batch predictions: {str(e)}")


# Custom exceptions (if not already defined elsewhere)
class PredictionError(Exception):
    """Raised when prediction generation fails"""
    pass


class InvalidTaskDataError(Exception):
    """Raised when task data validation fails"""
    pass