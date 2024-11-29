import logging

from mapof.core.distances import extract_distance_id

from mapof.marriages.distances import fast_distances as mrd
from mapof.marriages.distances.register import registered_marriages_distances
from mapof.marriages.objects.Marriages import Marriages


def get_distance(
        election_1: Marriages,
        election_2: Marriages,
        distance_id: str = None
) -> (float, list):
    """ Return: distance between ordinal elections, (if applicable) optimal matching """
    inner_distance, main_distance = extract_distance_id(distance_id)


    if main_distance in registered_marriages_distances:
        return registered_marriages_distances.get(main_distance)(election_1,
                                                                 election_2,
                                                                 inner_distance)
    else:
        logging.warning('No such metric!')

