from typing import Dict , Any , List
import logging
from src.services.model_service import ModelService, PredictionResult
from src.core.config import AppConfig
from src.services.task_extractor import TaskContextExtractor
from src.services.model_service import ModelService, PredictionResult
from src.core.config import AppConfig
from src.services.advice_strategy import ADVICE_STRATEGIES

logger = logging.getLogger(__name__)

class TaskAdviceService:
    """Generates actionable advice for tasks"""

    def __init__(self , model_service: ModelService , config: AppConfig = None):
        self.model_service = model_service
        self.context_extractor = TaskContextExtractor()
        self.config = config or AppConfig()

    def get_comprehensive_advice(self , task_data: Dict[str , Any]) -> Dict[str , Any]:
        """Generate comprehensive advice for a task"""

        try:
            context = self.context_extractor.extract(task_data)

            if context["is_completed"]:
                return self._generate_completed_task_response(context)

            prediction = self.model_service.predict(task_data)

            strategy = ADVICE_STRATEGIES.get(context["task_type"] , ADVICE_STRATEGIES["default"])
            advice_content = strategy.generate(context , prediction.__dict__)
            priority_level = (prediction.priority_prediction or "medium").capitalize()

            final_response = {
                'status_prediction':prediction.priority_prediction or "Medium" ,
                'confidence_score':prediction.priority_confidence or 0.5 ,
                'priority_recommendation':{
                    'level':priority_level ,
                    'estimated_days':prediction.duration_prediction or 1.0 ,
                    'message':self._get_priority_message(priority_level , context)
                } ,
                'context':context ,
                **advice_content
            }
            return final_response

        except Exception as e:
            logger.error(f"Generating advice failed: {e}" , exc_info=True)
            return { "error":"An internal error occurred when generating the advice." }

    def _get_priority_message(self, priority_level: str, context: Dict) -> str:
        task_type_str = context.get('task_type', 'general').replace('_', ' ')
        if priority_level == 'high':
            return f"Critical task of type'{task_type_str}'. Needs imediate attention."
        if priority_level == 'medium':
            return f"Important task of type '{task_type_str}'. Shouldbe planned soon."
        return f"Standard task of type '{task_type_str}'. Can be flexibly planned."


    def _generate_completed_task_response(self, context: Dict) -> Dict[str, Any]:
        """Returnează un răspuns standardizat pentru sarcinile finalizate."""
        return {
            "is_completed": True,
            "status_prediction":"Completed" ,
            "confidence_score":1.0 ,
            "message": "AI advice is not generated for tasks that are already completed.",

            "actionable_suggestions": [
                "Consider archiving this task to keep your workspace organized.",
                "Ensure all outcomes and lessons learned have been properly documented."
            ],
            "risk_factors":[] ,
            "optimization_tips":[
                "Review this task's history to create templates for similar future work."
            ] ,
            "priority_recommendation":{
                'level':'completed' ,
                'estimated_days':0 ,
                'message':"This task is already completed. No further action is required."
            } ,

            "next_steps":[
                "Notify all relevant stakeholders of the completion." ,
                "Finalize and store all related documentation."
            ] ,
            "context": context

        }