# Some distances are added via decorators
registered_marriages_distances = {
}

def register_marriages_distance(feature_id: str):

    def decorator(func):
        registered_marriages_distances[feature_id] = func
        return func

    return decorator
