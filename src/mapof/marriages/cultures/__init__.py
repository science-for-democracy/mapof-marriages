import logging

import numpy as np

import mapof.marriages.cultures.euclidean as euclidean
import mapof.marriages.cultures.impartial as impartial
import mapof.marriages.cultures.mallows as mallows
import mapof.marriages.cultures.urn as urn

independent_models = {
    'ic': impartial.generate_ic_votes,
    'id': impartial.generate_id_votes,
    'symmetric': impartial.generate_symmetric_votes,
    'norm-mallows': mallows.generate_norm_mallows_votes,
    'urn': urn.generate_urn_votes,
    'group_ic': impartial.generate_group_ic_votes,
}

dependent_models = {
    'malasym': mallows.generate_mallows_asymmetric_votes,
    'asymmetric': impartial.generate_asymmetric_votes,
    'euclidean': euclidean.generate_euclidean_votes,
    'reverse_euclidean': euclidean.generate_reverse_euclidean_votes,
    'mallows_euclidean': euclidean.generate_mallows_euclidean_votes,
    'expectation': euclidean.generate_expectation_votes,
    'attributes': euclidean.generate_attributes_votes,
    'fame': euclidean.generate_fame_votes,
}


def generate_votes(
        culture_id: str = None,
        num_agents: int = None,
        params: dict = None
) -> list | np.ndarray:

    if culture_id in independent_models:
        votes_1 = independent_models.get(culture_id)(num_agents=num_agents, **params)
        votes_2 = independent_models.get(culture_id)(num_agents=num_agents, **params)
        return [votes_1, votes_2]

    elif culture_id in dependent_models:
        return dependent_models.get(culture_id)(num_agents=num_agents, **params)

    else:
        logging.warning(f'No such culture id: {culture_id}')
        return []