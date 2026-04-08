TASKS = {
    "easy": {
        "size": 4,
        "start": [0, 0],
        "goal": [3, 3]
    },
    "medium": {
        "size": 6,
        "start": [0, 0],
        "goal": [5, 5]
    },
    "hard": {
        "size": 8,
        "start": [0, 0],
        "goal": [7, 7]
    }
}


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
