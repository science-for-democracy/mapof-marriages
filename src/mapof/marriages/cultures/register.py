
# Some cultures are added via decorators
registered_marriages_independent_cultures = {
}

registered_marriages_dependent_cultures = {
}


def register_marriages_independent_culture(feature_id: str):

    def decorator(func):
        registered_marriages_independent_cultures[feature_id] = func
        return func

    return decorator


def register_marriages_dependent_culture(feature_id: str):

    def decorator(func):
        registered_marriages_dependent_cultures[feature_id] = func
        return func

    return decorator
