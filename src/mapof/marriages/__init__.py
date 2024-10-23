
from .objects.MarriagesExperiment import MarriagesExperiment
from .objects.Marriages import Marriages
from .cultures import generate_votes
from .distances import get_distance


def prepare_online_marriages_experiment(**kwargs):
    return prepare_marriages_experiment(is_exported=False, is_imported=False, **kwargs)


def prepare_offline_marriages_experiment(**kwargs):
    return prepare_marriages_experiment(is_exported=True, is_imported=True, **kwargs)


def prepare_marriages_experiment(**kwargs):
    return MarriagesExperiment(**kwargs)


# def prepare_marriages_experiment(
#         experiment_id=None,
#         instances=None,
#         distances=None,
#         instance_type=None,
#         coordinates=None,
#         distance_id=None,
#         is_imported=False,
#         is_exported=True,
#         coordinates_names=None,
#         embedding_id=None,
#         fast_import=False,
#         with_matrix=False,
#         dim=2
# ):
#     return MarriagesExperiment(
#             experiment_id=experiment_id,
#             instances=instances,
#             is_exported=is_exported,
#             is_imported=is_imported,
#             distances=distances,
#             coordinates=coordinates,
#             distance_id=distance_id,
#             coordinates_names=coordinates_names,
#             embedding_id=embedding_id,
#             fast_import=fast_import,
#             with_matrix=with_matrix,
#             instance_type=instance_type,
#             dim=dim
#     )


def generate_marriages_instance(**kwargs):
    instance = Marriages('virtual', 'tmp', **kwargs)
    instance.prepare_instance()
    return instance


def generate_marriages_votes(**kwargs):
    return generate_votes(**kwargs)


def compute_distance(*args, **kwargs):
    return get_distance(*args, **kwargs)