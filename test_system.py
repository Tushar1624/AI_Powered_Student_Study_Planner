import joblib

print("Testing Models...")

performance_model = joblib.load(
    "models/performance_model.pkl"
)

priority_model = joblib.load(
    "models/priority_model.pkl"
)

print("✓ Models Loaded")

from planner.planner_utils import (
    create_study_plan
)

print("✓ Planner Imported")

from AI.ai_recommendation import (
    analyze_student
)

print("✓ AI Imported")

print("\nAll Tests Passed")