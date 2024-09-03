"""
function to validate input data, if it includes all required fields and has the correct format and data types


Input data format:
[
  {
    "userId": "user1",
    "data": {
      "day21": {
        "numberSessions": 3,
        "kmTotal": 12.5,
        "kmZ3Z4": 4.5,
        "kmZ5": 1.2,
        "kmSprint": 0.8,
        "hoursAlternative": 2,
        "numberStrengthSessions": 2,
        "perceivedTrainingSuccess": 8,
        "perceivedRecovery": 6,
        "perceivedExertion": 7,
        "injured": false
      }, ...
      }, ...}
  }]
"""
def validate_input(input_data, days=56):

    required_fields = {
        "numberSessions": (int, float),
        "kmTotal": (int, float),
        "kmZ3Z4": (int, float),
        "kmZ5": (int, float),
        "kmSprint": (int, float),
        "hoursAlternative": (int, float),
        "numberStrengthSessions": (int, float),
        "perceivedTrainingSuccess": (int, float),
        "perceivedRecovery": (int, float),
        "perceivedExertion": (int, float),
        "injured": bool
    }

    if not isinstance(input_data, list):
        print("Input data is not a list.")
        return False

    for user_data in input_data:
        if not isinstance(user_data, dict):
            return False
        if "userId" not in user_data or not isinstance(user_data["userId"], str):
            return False
        if "data" not in user_data or not isinstance(user_data["data"], dict):
            return False

        for day in range(1, days + 1):
            day_key = f"day{day}"
            if day_key not in user_data["data"]:
                print(f"Day {day} not found for user {user_data['userId']}")
                return False
            day_data = user_data["data"][day_key]
            if not isinstance(day_data, dict):
                print(f"Day {day} data is not a dictionary for user {user_data['userId']}")
                return False

            for field, field_type in required_fields.items():
                if field not in day_data:
                    print(f"Field {field} not found for day {day} for user {user_data['userId']}")
                    return False
                if not isinstance(day_data[field], field_type):
                    print(f"Field {field} has the wrong type for day {day} for user {user_data['userId']}")
                    return False

    return True