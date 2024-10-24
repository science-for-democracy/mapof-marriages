import copy
import csv
import os
from time import time
from typing import Callable

import mapof.core.persistence.experiment_exports as exports
import numpy as np
from mapof.core.inner_distances import map_str_to_func
from mapof.core.objects.Experiment import Experiment

from mapof.marriages.distances import main_marriages_distances as mrd
from mapof.marriages.objects.Marriages import Marriages


def get_distance(election_1: Marriages, election_2: Marriages,
                 distance_id: str = None) -> (float, list):
    """ Return: distance between ordinal elections, (if applicable) optimal matching """
    inner_distance, main_distance = extract_distance_id(distance_id)

    metrics_without_params = {

    }

    metrics_with_inner_distance = {
        'mutual_attraction': mrd.compute_retrospective_distance,
    }

    if main_distance in metrics_without_params:
        return metrics_without_params.get(main_distance)(election_1, election_2)

    elif main_distance in metrics_with_inner_distance:
        return metrics_with_inner_distance.get(main_distance)(election_1, election_2,
                                                              inner_distance)


def extract_distance_id(distance_id: str) -> (Callable, str):
    if '-' in distance_id:
        inner_distance, main_distance = distance_id.split('-')
        inner_distance = map_str_to_func(inner_distance)
    else:
        main_distance = distance_id
        inner_distance = None
    return inner_distance, main_distance


def run_single_process(exp: Experiment, instances_ids: list,
                      distances: dict, times: dict, matchings: dict,
                      printing: bool) -> None:
    """ Single thread for computing distances """

    for instance_id_1, instance_id_2 in instances_ids:

        start_time = time()
        distance = get_distance(copy.deepcopy(exp.instances[instance_id_1]),
                                copy.deepcopy(exp.instances[instance_id_2]),
                                distance_id=copy.deepcopy(exp.distance_id))
        print(distance)
        if type(distance) is tuple:
            distance, matching = distance
            matching = np.array(matching)
            matchings[instance_id_1][instance_id_2] = matching
            matchings[instance_id_2][instance_id_1] = np.argsort(matching)
        distances[instance_id_1][instance_id_2] = distance
        distances[instance_id_2][instance_id_1] = distances[instance_id_1][instance_id_2]
        times[instance_id_1][instance_id_2] = time() - start_time
        times[instance_id_2][instance_id_1] = times[instance_id_1][instance_id_2]


def run_multiple_processes(
        experiment: Experiment,
        instances_ids: list,
        distances: dict,
        times: dict,
        matchings: dict,
        process_id: int
) -> None:
    """ Single thread for computing distances """

    for instance_id_1, instance_id_2 in instances_ids:
        start_time = time()
        distance = get_distance(copy.deepcopy(experiment.instances[instance_id_1]),
                                copy.deepcopy(experiment.instances[instance_id_2]),
                                distance_id=copy.deepcopy(experiment.distance_id))
        if type(distance) is tuple:
            distance, matching = distance
            matching = np.array(matching)
            matchings[instance_id_1][instance_id_2] = matching
            matchings[instance_id_2][instance_id_1] = np.argsort(matching)
        distances[instance_id_1][instance_id_2] = distance
        distances[instance_id_2][instance_id_1] = distances[instance_id_1][instance_id_2]
        times[instance_id_1][instance_id_2] = time() - start_time
        times[instance_id_2][instance_id_1] = times[instance_id_1][instance_id_2]

    if experiment.is_exported:
        exports.export_distances_multiple_processes(experiment,
                                                    instances_ids,
                                                    distances,
                                                    times,
                                                    process_id)

