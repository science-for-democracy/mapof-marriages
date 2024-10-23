
from .objects.MarriagesExperiment import MarriagesExperiment
from .objects.Marriages import Marriages
from .cultures import generate_votes
from .distances import get_distance


def prepare_online_marriages_experiment(**kwargs):
    return prepare_marriages_experiment(**kwargs, is_exported=False, is_imported=False)


def prepare_offline_marriages_experiment(**kwargs):
    return prepare_marriages_experiment(**kwargs, is_exported=True, is_imported=True)


def prepare_marriages_experiment(**kwargs):

    return MarriagesExperiment(
        **kwargs
    )


def generate_marriages_instance(**kwargs):
    instance = Marriages('virtual', 'tmp', **kwargs)
    instance.prepare_instance()
    return instance


def generate_marriages_votes(**kwargs):
    return generate_votes(**kwargs)


def compute_distance(*args, **kwargs):
    return get_distance(*args, **kwargs)