# Some features are added via decorators
registered_marriages_features = {
}

def register_marriages_feature(feature_id: str):

    def decorator(func):
        registered_marriages_features[feature_id] = func
        return func

    return decorator
