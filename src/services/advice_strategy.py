from typing import Dict , List
from src.core.data_config import scenarios


class BaseAdviceStrategy:

    def generate(self , context: Dict , prediction: Dict) -> Dict:
        return {
            "actionable_suggestions":self._generate_recommendations(context , prediction) ,
            "risk_factors":self._generate_risk_factors(context , prediction) ,
            "industry_insights":self._generate_industry_insights(context),
            "optimization_tips":[],
            "next_steps":[]
        }

    def _generate_recommendations(self , context: Dict , prediction: Dict) -> List[str]:
        recs = ["Review and clarify the task's objectives and success criteria before starting."]
        print(prediction.get('duration_prediction'))

        if prediction.get('duration_prediction' , 0) > 30:
            recs.append("For long-duration tasks, break down the work into clear milestones.")
        return recs

    def _generate_risk_factors(self , context: Dict , prediction: Dict) -> List[str]:
        return ["The primary risk is a lack of clarity in requirements, which can lead to rework."]

    def _generate_industry_insights(self , context: Dict) -> List[str]:
        industry = context.get("industry")
        if industry in scenarios:
            config = scenarios[industry]
            avg_duration = sum(config["average_duration_range"]) / 2
            stakeholders = ", ".join(config.get("stakeholder_types" , ["the project team"])[:2])
            return [
                f"In the '{industry.replace('_', ' ').title()}' industry, similar tasks take approximately {avg_duration:.0f} days.",
                f"Be prepared to collaborate with roles such as: {stakeholders}."
            ]
        return []


class RegulatoryComplianceStrategy(BaseAdviceStrategy):

    def _generate_recommendations(self , context: Dict , prediction: Dict) -> List[str]:
        return [
            "Rigorously document all applicable compliance requirements.",
            "Engage the legal department from the initial phases of the project.",
            "Establish regular checkpoints to validate ongoing compliance."
        ]

    def _generate_risk_factors(self , context: Dict , prediction: Dict) -> List[str]:
        risks = super()._generate_risk_factors(context , prediction)
        risks.extend([
            "Penalties for non-compliance can be significant and costly.",
            "Legislation may change during the project lifecycle, requiring adaptation.",
            "Obtaining approvals from regulatory authorities can take longer than estimated."
        ])
        return risks


class CrisisManagementStrategy(BaseAdviceStrategy):

    def _generate_recommendations(self , context: Dict , prediction: Dict) -> List[str]:
        return [
            "Immediately activate the established crisis management protocol.",
            "Establish a single, clear channel of communication for all stakeholders.",
            "Inform senior leadership and the communications department within the first hour."
        ]

    def _generate_risk_factors(self , context: Dict , prediction: Dict) -> List[str]:
        return [
            "Time pressure can lead to rushed decisions and overlooked details." ,
            "Unclear or inconsistent communication can worsen the crisis and damage reputation." ,
            "The emotional impact on the team can reduce efficiency and decision quality."
        ]


ADVICE_STRATEGIES = {
    "regulatory_compliance":RegulatoryComplianceStrategy() ,
    "crisis_management":CrisisManagementStrategy() ,
    "default":BaseAdviceStrategy()
    #more to be added if needed
}