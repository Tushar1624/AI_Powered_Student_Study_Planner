def generate_plan(subjects):

    plan = []

    total_hours = 8

    total_priority = sum(
        subject["priority_score"]
        for subject in subjects
    )

    for subject in subjects:

        hours = round(
            (subject["priority_score"] / total_priority)
            * total_hours,
            1
        )

        plan.append({
            "subject": subject["name"],
            "hours": hours
        })

    return plan