from AI.ai_recommendation import (
    analyze_student
)

student = {

    "study_hours": 3,
    "attendance": 70,
    "stress_level": 8,
    "motivation": 4,
    "predicted_score": 65
}

subject_scores = {

    "Math": 45,
    "Physics": 80,
    "Programming": 55
}

result = analyze_student(
    student,
    subject_scores
)

print(result)