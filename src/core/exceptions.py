class TaskModelError(Exception):
    """Base exception for task model operations"""
    pass

class DatabaseConnectionError(TaskModelError):
    """Raised when database connection fails"""
    pass

class ModelNotFoundError(TaskModelError):
    """Raised when model files are not found"""
    pass

class PredictionError(TaskModelError):
    """Raised when prediction fails"""
    pass

class DataProcessingError(TaskModelError):
    """Raised when data processing fails"""
    pass