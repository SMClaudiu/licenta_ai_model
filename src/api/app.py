# app.py
from flask import Flask
from flask_cors import CORS
from routes import create_routes
from src.models.model_registry import ModelRegistry
from src.services.advice_service import TaskAdviceService
from src.services.model_service import ModelService
from src.core.config import AppConfig
import logging


def create_app(config=None):
    app = Flask(__name__)

    # Enable CORS for frontend communication
    CORS(app, resources={
        r"/*": {"origins": ["http://localhost:3000", "http://localhost:5173"]}
    })

    # Load configuration
    if config:
        app.config.from_object(config)
    else:
        app.config.from_object(AppConfig())

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    app.logger.setLevel(logging.INFO)

    # Initialize services
    try:
        model_registry = ModelRegistry.get_instance()
        model_service = ModelService(model_registry)
        advice_service = TaskAdviceService(model_service)

        # Store services in app context for access in routes
        app.advice_service = advice_service
        app.model_service = model_service

        app.logger.info("Services initialized successfully")
    except Exception as e:
        app.logger.error(f"Failed to initialize services: {str(e)}")
        raise

    # Register blueprints
    app.register_blueprint(create_routes(advice_service))

    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        try:
            # Test model service availability
            is_healthy = model_service.is_healthy() if hasattr(model_service, 'is_healthy') else True
            return {
                'status': 'healthy' if is_healthy else 'degraded',
                'services': {
                    'model_service': 'operational' if is_healthy else 'degraded',
                    'advice_service': 'operational'
                }
            }, 200 if is_healthy else 503
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }, 500

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)