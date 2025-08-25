# src/services/task_context_extractor.py
import re
from typing import Dict , Any
from src.core.data_config import scenarios, user_profiles


INDUSTRY_INDICATORS = {
    key: scenario.get("current_issues", []) for key, scenario in scenarios.items()
}

TASK_TYPE_PATTERNS = {
    "regulatory_compliance": ["compliance", "regulatory", "audit", "legal", "approval", "filing"],
    "crisis_management": ["crisis", "emergency", "urgent", "critical", "escalated", "immediate"],
    "strategic_planning": ["strategy", "planning", "initiative", "roadmap", "vision", "long-term"],
    "technology_implementation": ["implement", "deploy", "integrate", "migrate", "upgrade", "technology"],
    "stakeholder_engagement": ["meeting", "stakeholder", "consultation", "review", "approval", "sign-off"],
    "research_analysis": ["research", "analysis", "study", "investigate", "assess", "evaluate"]
}
URGENCY_INDICATORS = {
    "URGENT:": 0.95, "CRITICAL:": 0.90, "HIGH PRIORITY:": 0.85,
    "ESCALATED:": 0.80, "IMMEDIATE ACTION:": 0.95
}

class TaskContextExtractor:

    def extract(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        task_name = task_data.get('name', '').lower()
        description = task_data.get('description', '').lower()
        status = task_data.get('status', 'Not_Done')
        combined_text = f"{task_name} {description}"

        is_completed = status in ['Done', 'Completed', 2]

        return {
            "is_completed": is_completed,
            "industry": self._detect_entity(combined_text, INDUSTRY_INDICATORS, "general"),
            "task_type": self._detect_entity(combined_text, TASK_TYPE_PATTERNS, "general_task"),
            "urgency": self._calculate_urgency(task_name),
            "company": self._extract_company(description),
            "user_profile": self._detect_user_profile(description)
        }

    def _detect_entity(self, text: str, patterns: Dict, default: str) -> str:
        scores = {key: sum(1 for keyword in keywords if keyword in text) for key, keywords in patterns.items()}
        scored_entities = {k: v for k, v in scores.items() if v > 0}
        return max(scored_entities, key=scored_entities.get) if scored_entities else default

    def _calculate_urgency(self, task_name: str) -> str:
        for indicator, score in URGENCY_INDICATORS.items():
            if indicator.lower() in task_name:
                if score >= 0.85: return "high"
                if score >= 0.70: return "medium"
        return "low"

    def _extract_company(self, description: str) -> str:
        match = re.search(r'Company: ([^|]+)', description, re.IGNORECASE)
        return match.group(1).strip() if match else "Unknown Company"

    def _detect_user_profile(self, description: str) -> str:
        desc_len = len(description)
        if desc_len >= user_profiles["executive_level"]["description_length"][0]: return "executive_level"
        if desc_len >= user_profiles["senior_level"]["description_length"][0]: return "senior_level"
        if desc_len >= user_profiles["mid_level"]["description_length"][0]: return "mid_level"
        return "entry_level"