from planner import generate_plan

subjects = [
    {"name": "AI", "priority_score": 9},
    {"name": "DBMS", "priority_score": 5},
    {"name": "Python", "priority_score": 2}
]

plan = generate_plan(subjects)

print(plan)