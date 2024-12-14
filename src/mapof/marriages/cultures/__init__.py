import logging

import numpy as np

import mapof.marriages.cultures.euclidean as euclidean
import mapof.marriages.cultures.impartial as impartial
import mapof.marriages.cultures.mallows as mallows
import mapof.marriages.cultures.urn as urn


from mapof.marriages.cultures.register import (
    registered_marriages_independent_cultures,
    registered_marriages_dependent_cultures,
)


def generate_votes(
        culture_id: str = None,
        num_agents: int = None,
        params: dict = None
) -> list | np.ndarray:

    if culture_id in registered_marriages_independent_cultures:
        votes_1 = registered_marriages_independent_cultures.get(culture_id)(
            num_agents=num_agents, **params)
        votes_2 = registered_marriages_independent_cultures.get(culture_id)(
            num_agents=num_agents, **params)
        return [votes_1, votes_2]

    elif culture_id in registered_marriages_dependent_cultures:
        return registered_marriages_dependent_cultures.get(culture_id)(
            num_agents=num_agents, **params)

    else:
        logging.warning(f'No such culture id: {culture_id}')
        return []