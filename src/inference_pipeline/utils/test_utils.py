import pytest
import numpy as np
from preprocess import standardise, normalise, reshape

def test_standardise():
    input_data = [
        {
            "userId": "user1",
            "data": {
                f"day{day}": {
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
                    "injured": False
                } for day in range(1, 22)
            }
        }
    ]

    mean_std_values = {
        "user1": {
            "numberSessions": {"mean": 2, "stdv": 1},
            "kmTotal": {"mean": 10, "stdv": 2},
            "kmZ3Z4": {"mean": 4, "stdv": 1},
            "kmZ5": {"mean": 1, "stdv": 0.5},
            "kmSprint": {"mean": 0.5, "stdv": 0.2},
            "hoursAlternative": {"mean": 1.5, "stdv": 0.5},
            "numberStrengthSessions": {"mean": 1.5, "stdv": 0.5},
            "perceivedTrainingSuccess": {"mean": 7, "stdv": 1},
            "perceivedRecovery": {"mean": 5, "stdv": 1},
            "perceivedExertion": {"mean": 6, "stdv": 0.5}
        }
    }

    expected_output = [
        {
            "userId": "user1",
            "data": {
                f"day{day}": {
                    "numberSessions": 1.0,
                    "kmTotal": 1.25,
                    "kmZ3Z4": 0.5,
                    "kmZ5": 0.4,
                    "kmSprint": 1.5,
                    "hoursAlternative": 1.0,
                    "numberStrengthSessions": 1.0,
                    "perceivedTrainingSuccess": 1.0,
                    "perceivedRecovery": 1.0,
                    "perceivedExertion": 2.0,
                    "injured": False
                } for day in range(1, 22)
            }
        }
    ]

    result = standardise(input_data, mean_std_values)
    for day in range(1, 22):
        day_key = f'day{day}'
        for key in expected_output[0]['data'][day_key]:
            assert round(result[0]['data'][day_key][key], 2) == round(expected_output[0]['data'][day_key][key], 2), f"Standardise function failed on {key} for {day_key}"

def test_normalise():
    input_data = [
        {
            "userId": "user1",
            "data": {
                f"day{day}": {
                    "numberSessions": 1.0,
                    "kmTotal": 1.25,
                    "kmZ3Z4": 0.5,
                    "kmZ5": 0.4,
                    "kmSprint": 1.5,
                    "hoursAlternative": 1.0,
                    "numberStrengthSessions": 1.0,
                    "perceivedTrainingSuccess": 1.0,
                    "perceivedRecovery": 1.0,
                    "perceivedExertion": 2.0,
                    "injured": False
                } for day in range(1, 22)
            }
        }
    ]

    min_max_values = {
        "numberSessions": {"min": -2.9536942523474603, "max": 5.107539184552492},
        "kmTotal": {"min": -2.0250511089018715, "max": 9.856641839539968},
        "kmZ3Z4": {"min": -0.5734425513790303, "max": 14.298993695957575},
        "kmZ5": {"min": -0.5917629229252217, "max": 29.908335116948038},
        "kmSprint": {"min": -0.7239681364071932, "max": 42.31872625574813},
        "numberStrengthSessions": {"min": -0.5761246407648938, "max": 24.939927826679853},
        "hoursAlternative": {"min": -0.5993841662903084, "max": 16.854295552877502},
        "perceivedExertion": {"min": -2.3182036603206555, "max": 11.995560702588447},
        "perceivedTrainingSuccess": {"min": -3.630210005420229, "max": 17.110288534188804},
        "perceivedRecovery": {"min": -3.424587935617414, "max": 9.142316049704705}
    }

    expected_output = [
        {
            "userId": "user1",
            "data": {
                f"day{day}": {
                    "numberSessions": round((1.0 - (-2.9536942523474603)) / (5.107539184552492 - (-2.9536942523474603)), 2),
                    "kmTotal": round((1.25 - (-2.0250511089018715)) / (9.856641839539968 - (-2.0250511089018715)), 2),
                    "kmZ3Z4": round((0.5 - (-0.5734425513790303)) / (14.298993695957575 - (-0.5734425513790303)), 2),
                    "kmZ5": round((0.4 - (-0.5917629229252217)) / (29.908335116948038 - (-0.5917629229252217)), 2),
                    "kmSprint": round((1.5 - (-0.7239681364071932)) / (42.31872625574813 - (-0.7239681364071932)), 2),
                    "hoursAlternative": round((1.0 - (-0.5993841662903084)) / (16.854295552877502 - (-0.5993841662903084)), 2),
                    "numberStrengthSessions": round((1.0 - (-0.5761246407648938)) / (24.939927826679853 - (-0.5761246407648938)), 2),
                    "perceivedTrainingSuccess": round((1.0 - (-3.630210005420229)) / (17.110288534188804 - (-3.630210005420229)), 2),
                    "perceivedRecovery": round((1.0 - (-3.424587935617414)) / (9.142316049704705 - (-3.424587935617414)), 2),
                    "perceivedExertion": round((2.0 - (-2.3182036603206555)) / (11.995560702588447 - (-2.3182036603206555)), 2),
                    "injured": False
                } for day in range(1, 22)
            }
        }
    ]

    result = normalise(input_data, min_max_values)
    for day in range(1, 22):
        day_key = f'day{day}'
        for key in expected_output[0]['data'][day_key]:
            if isinstance(result[0]['data'][day_key][key], float):
                assert round(result[0]['data'][day_key][key], 2) == round(expected_output[0]['data'][day_key][key], 2), f"Standardise function failed on {key} for {day_key}"
            else:
                assert result[0]['data'][day_key][key] == expected_output[0]['data'][day_key][key], f"Standardise function failed on {key} for {day_key}"


def test_reshape():
    input_data = [
        {
            "userId": "user1",
            "data": {
                f"day{day}": {
                    "numberSessions": 0.5363941443405939,
                    "kmTotal": 0.3228930414460134,
                    "kmZ3Z4": 0.25757639383873886,
                    "kmZ5": 0.06693972804362414,
                    "kmSprint": 0.05263157894736842,
                    "hoursAlternative": 0.094933272623934,
                    "numberStrengthSessions": 0.06611831192150868,
                    "perceivedTrainingSuccess": 0.24086513403266284,
                    "perceivedRecovery": 0.303987282002244,
                    "perceivedExertion": 0.15346145533816768,
                    "injured": False
                } for day in range(1, 22)
            }
        }
    ]

    expected_output = np.array([
        [
            [
                0.5363941443405939, 0.3228930414460134, 0.25757639383873886, 0.06693972804362414, 0.05263157894736842,
                0.06611831192150868, 0.094933272623934, 0.15346145533816768, 0.303987282002244, 0.24086513403266284
            ] for day in range(1, 57)
        ]
    ])

    result = reshape(input_data)
    assert np.array_equal(result, expected_output), "Reshape function failed"