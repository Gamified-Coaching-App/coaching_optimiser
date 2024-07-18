def format_output(output, user_ids):
    feature_order = [
        "numberSessions",
        "kmTotal",
        "kmZ3Z4",
        "kmZ5",
        "kmSprint",
        "hoursAlternative",
        "numberStrengthSessions"
    ]
    
    formatted_output = {}
    for i, user_id in enumerate(user_ids):
        formatted_output[user_id] = {}
        for day in range(output.shape[1]):
            day_key = f"day{day + 1}"
            formatted_output[user_id][day_key] = {feature: float(output[i, day, idx]) for idx, feature in enumerate(feature_order)}
    
    return formatted_output