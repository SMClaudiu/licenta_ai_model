import numpy as np
import pandas as pd
from datetime import datetime , timedelta
from typing import Dict , List , Tuple


class TaskAdviceGenerator:
    def __init__(self , classification_processor , regression_processor ,
                 classifier_trainer , regressor_trainer):
        self.classification_processor = classification_processor
        self.regression_processor = regression_processor
        self.classifier_trainer = classifier_trainer
        self.regressor_trainer = regressor_trainer

        # Define advice templates
        self.status_advice = {
            0:"pending_advice" ,
            1:"in_progress_advice" ,
            2:"completed_advice"
        }

        self.urgency_thresholds = {
            'high':3 ,
            'medium':7 ,
            'low':14
        }

    def generate_advice(self , task_data: Dict) -> Dict:
        """
        Generate comprehensive advice for a single task
        """
        try:
            # Prepare task data for prediction
            processed_data = self._prepare_single_task(task_data)

            # Get predictions
            status_pred , status_confidence = self._predict_status(processed_data)
            completion_time = self._predict_completion_time(processed_data)

            # Generate advice components
            advice = {
                'status_prediction':self._get_status_label(status_pred) ,
                'confidence_score':float(status_confidence) ,
                'estimated_completion':float(completion_time) ,
                'priority_recommendation':self._get_priority_advice(completion_time , task_data) ,
                'actionable_suggestions':self._get_actionable_suggestions(status_pred , completion_time , task_data) ,
                'risk_factors':self._identify_risk_factors(task_data , completion_time) ,
                'optimization_tips':self._get_optimization_tips(task_data , completion_time) ,
                'next_steps':self._generate_next_steps(status_pred , completion_time , task_data)
            }

            return {
                'success':True ,
                'advice':advice ,
                'metadata':{
                    'generated_at':datetime.now().isoformat() ,
                    'model_version':'1.0'
                }
            }

        except Exception as e:
            return {
                'success':False ,
                'error':str(e) ,
                'advice':self._get_fallback_advice(task_data)
            }

    def _prepare_single_task(self , task_data: Dict) -> Tuple[np.ndarray , np.ndarray]:
        """Convert single task data to model input format"""
        # Create DataFrame with single task
        df = pd.DataFrame([{
            'task_name':task_data.get('name' , '') ,
            'description':task_data.get('description' , '') ,
            'creation_date':pd.to_datetime(task_data.get('creation_date' , datetime.now())) ,
            'due_date':pd.to_datetime(task_data.get('due_date' , datetime.now() + timedelta(days=7))) ,
            'board_name':task_data.get('board_name' , 'Default') ,
            'client_name':task_data.get('client_name' , 'Unknown') ,
            'email':task_data.get('client_email' , 'unknown@example.com') ,
            # Add default values for required fields
            **self._get_default_values()
        }])

        # Process for classification
        df_class = self.classification_processor.engineer_advanced_features(df.copy() , fit_mode=False)
        class_features = df_class[self.classification_processor.feature_names].fillna(0)
        scaled_class_features = self.classification_processor.scaler.transform(class_features)

        # Process for regression
        df_reg = self.regression_processor.engineer_advanced_features(df.copy() , fit_mode=False)
        reg_features = df_reg[self.regression_processor.feature_names].fillna(0)
        scaled_reg_features = self.regression_processor.scaler.transform(reg_features)

        return scaled_class_features , scaled_reg_features

    def _get_default_values(self) -> Dict:
        """Get default values for missing task attributes"""
        return { col:avg for col , avg in self.regression_processor.global_averages.items() }

    def _predict_status(self , processed_data: Tuple) -> Tuple[int , float]:
        """Predict task status with confidence"""
        class_features = processed_data[0]
        predictions , _ , confidence = self.classifier_trainer.predict(
            class_features , return_confidence=True
        )
        return int(predictions[0]) , float(confidence[0])

    def _predict_completion_time(self , processed_data: Tuple) -> float:
        """Predict task completion time"""
        reg_features = processed_data[1]
        time_pred = self.regressor_trainer.predict(reg_features)
        return max(1.0 , float(time_pred[0]))  # Minimum 1 day

    def _get_status_label(self , status_pred: int) -> str:
        """Convert status prediction to human-readable label"""
        labels = { 0:'Pending' , 1:'In Progress' , 2:'Completed' }
        return labels.get(status_pred , 'Unknown')

    def _get_priority_advice(self , completion_time: float , task_data: Dict) -> Dict:
        """Generate priority-based advice"""
        if completion_time <= self.urgency_thresholds['high']:
            priority = 'high'
            message = "High Priority: This task requires immediate attention"
        elif completion_time <= self.urgency_thresholds['medium']:
            priority = 'medium'
            message = "Medium Priority: Schedule this task within the next few days"
        else:
            priority = 'low'
            message = "Low Priority: This task can be scheduled flexibly"

        return {
            'level':priority ,
            'message':message ,
            'estimated_days':completion_time
        }

    def _get_actionable_suggestions(self , status_pred: int , completion_time: float , task_data: Dict) -> List[str]:
        """Generate specific actionable suggestions"""
        suggestions = []

        # Status-based suggestions
        if status_pred == 0:  # Pending
            suggestions.append("Break down this task into smaller, manageable subtasks")
            suggestions.append("Set a specific start date to avoid procrastination")
        elif status_pred == 1:  # In Progress
            suggestions.append("Review current progress and adjust timeline if needed")
            suggestions.append("Focus on completing this task before starting new ones")

        # Time-based suggestions
        if completion_time <= 2:
            suggestions.append("Consider this urgent - allocate dedicated time blocks")
            suggestions.append("Inform team members about the tight deadline")
        elif completion_time > 20:
            suggestions.append("Create milestone checkpoints to track long-term progress")
            suggestions.append("Consider if this task can be simplified or delegated")

        # Description-based suggestions
        description = task_data.get('description' , '').lower()
        if any(keyword in description for keyword in ['bug' , 'fix' , 'error']):
            suggestions.append("ocument the issue thoroughly before starting the fix")
        if any(keyword in description for keyword in ['meeting' , 'review' , 'approve']):
            suggestions.append("Send calendar invites and prepare agenda items in advance")

        return suggestions[:4]  # Limit to 4 most relevant suggestions

    def _identify_risk_factors(self , task_data: Dict , completion_time: float) -> List[str]:
        """Identify potential risk factors"""
        risks = []

        # Time-based risks
        due_date = pd.to_datetime(task_data.get('due_date'))
        days_until_due = (due_date - datetime.now()).days

        if days_until_due < completion_time:
            risks.append("Timeline Risk: Estimated completion exceeds deadline")

        if completion_time > 30:
            risks.append("Scope Risk: Long-duration tasks often face scope creep")

        # Weekend/holiday risks
        creation_date = pd.to_datetime(task_data.get('creation_date' , datetime.now()))
        if creation_date.weekday() >= 5:  # Weekend
            risks.append("Timing Risk: Created on weekend - may have delayed start")

        # Description-based risks
        description = task_data.get('description' , '').lower()
        if len(description) < 20:
            risks.append("Clarity Risk: Task description is too brief - may cause confusion")

        return risks[:3]  # Limit to 3 most critical risks

    def _get_optimization_tips(self , task_data: Dict , completion_time: float) -> List[str]:
        """Generate optimization tips"""
        tips = []

        # General optimization tips based on completion time
        if completion_time <= 3:
            tips.append("Use time-blocking technique - dedicate 2-3 hour focused sessions")
            tips.append("Turn off notifications during work sessions")
        elif completion_time <= 7:
            tips.append("Create daily mini-goals to maintain momentum")
            tips.append("Use the Pomodoro Technique for sustained focus")
        else:
            tips.append("Allocate 15-20% buffer time for unexpected complications")
            tips.append("Set weekly checkpoints to review progress")

        # Task-specific optimization
        description = task_data.get('description' , '').lower()
        if any(keyword in description for keyword in ['research' , 'analysis']):
            tips.append("Start with a quick literature review to avoid duplicating work")
        if any(keyword in description for keyword in ['design' , 'create']):
            tips.append("Begin with rough sketches/wireframes before detailed work")

        return tips[:3]  # Limit to 3 most actionable tips

    def _generate_next_steps(self , status_pred: int , completion_time: float , task_data: Dict) -> List[str]:
        """Generate concrete next steps"""
        steps = []

        if status_pred == 0:  # Pending
            steps.append("Review task requirements and clarify any ambiguities")
            steps.append("Gather necessary resources and tools")
            steps.append("Block time in calendar for focused work sessions")
        elif status_pred == 1:  # In Progress
            steps.append("Assess current progress and remaining work")
            steps.append("Update stakeholders on status and timeline")
            steps.append("Identify any blockers and create resolution plan")
        else:  # Completed or nearly complete
            steps.append("Conduct final quality review")
            steps.append("Prepare handover documentation")
            steps.append("Schedule follow-up with stakeholders")

        return steps[:3]

    def _get_fallback_advice(self , task_data: Dict) -> Dict:
        """Provide basic advice when AI prediction fails"""
        return {
            'status_prediction':'Unknown' ,
            'confidence_score':0.0 ,
            'estimated_completion':5.0 ,
            'priority_recommendation':{
                'level':'medium' ,
                'message':'Unable to determine priority - please review manually'
            } ,
            'actionable_suggestions':[
                "Review task description and add more details" ,
                "Set a realistic deadline if not already specified" ,
                "Break down into smaller subtasks for better tracking"
            ] ,
            'risk_factors':["⚠Insufficient data for risk analysis"] ,
            'optimization_tips':["Ensure clear requirements before starting"] ,
            'next_steps':[
                "Clarify task requirements and expectations" ,
                "Set preliminary timeline and milestones" ,
                "Begin with research/planning phase"
            ]
        }