# routes.py
from flask import Blueprint, request, jsonify, current_app
from datetime import datetime


def create_routes(advice_service):
    api = Blueprint('api', __name__)

    @api.route('/predict/advice', methods=['POST'])
    def get_task_advice():

        # Accepts TaskAdviceRequest and returns TaskAdviceResponse

        try:
            task_data = request.json

            if not task_data:
                return jsonify({
                    'success': False,
                    'error': 'Request body is required'
                }), 400

            required_fields = ['id', 'name', 'description']
            missing_fields = [field for field in required_fields if not task_data.get(field)]
            current_app.logger.info(f"Received task data for prediction: {task_data}")

            if missing_fields:
                return jsonify({
                    'success': False,
                    'error': f'Missing required fields: {", ".join(missing_fields)}'
                }), 400

            advice_data = advice_service.get_comprehensive_advice(task_data)

            # Format response to match frontend expectations
            response = {
                'success': True,
                'advice': advice_data,
                'metadata': {
                    'generated_at': datetime.utcnow().isoformat() + 'Z',
                    'model_version': '1.0.0'
                }
            }
            current_app.logger.info("This is the advice requested" , advice_data)
            current_app.logger.info(f"Generated advice for task {task_data.get('id')}")
            return jsonify(response)

        except ValueError as e:
            current_app.logger.warning(f"Invalid input data: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Invalid input: {str(e)}'
            }), 400

        except Exception as e:
            current_app.logger.error(f"Error generating advice: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Internal server error occurred while generating advice'
            }), 500

    @api.route('/predict/status', methods=['POST'])
    def predict_task_status():

        # Endpoint for status prediction only

        try:
            task_data = request.json

            if not task_data:
                return jsonify({
                    'success': False,
                    'error': 'Request body is required'
                }), 400

            prediction = advice_service.predict_status_only(task_data)

            return jsonify({
                'success': True,
                'prediction': prediction,
                'metadata': {
                    'generated_at': datetime.utcnow().isoformat() + 'Z'
                }
            })

        except Exception as e:
            current_app.logger.error(f"Error predicting status: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Internal server error occurred during prediction'
            }), 500

    @api.route('/health', methods=['GET'])
    def service_health():

        # Detailed health check for the advice service

        try:
            health_status = advice_service.get_health_status()
            return jsonify(health_status)
        except Exception as e:
            return jsonify({
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }), 500

    return api