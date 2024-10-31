from mapof.core.matchings import *


def compute_retrospective_distance(
        instance_1,
        instance_2,
        inner_distance
) -> (float, list | None):
    """ Computes Retrospective distance between marriages instances

    Parameters
    ----------
        instance_1 : Marriage
            First marriage instance.
        instance_2 : Marriage
            Second marriage instance.
        inner_distance : callable
            Inner distance.

    Returns
    -------
        float
            Retrospective distance.
    """
    results = []
    for crossing in [False, True]:
        cost_table_1, cost_table_2 = _get_matching_cost_retrospective(
            instance_1, instance_2, inner_distance, crossing=crossing)

        a, _ = solve_matching_vectors(cost_table_1)
        b, _ = solve_matching_vectors(cost_table_2)
        results.append(a+b)

    return min(results), None


def _get_matching_cost_retrospective(
        instance_1,
        instance_2,
        inner_distance: callable,
        crossing=False
):
    """ Return: Cost table """
    vectors_1 = instance_1.get_retrospective_vectors()
    vectors_2 = instance_2.get_retrospective_vectors()

    size = instance_1.num_agents

    if crossing:
        return [[inner_distance(vectors_1[0][i], vectors_2[1][j])
                 for i in range(size)] for j in range(size)], \
               [[inner_distance(vectors_1[1][i], vectors_2[0][j])
                 for i in range(size)] for j in range(size)]

    else:
        return [[inner_distance(vectors_1[0][i], vectors_2[0][j])
                 for i in range(size)] for j in range(size)], \
               [[inner_distance(vectors_1[1][i], vectors_2[1][j])
                 for i in range(size)] for j in range(size)]
