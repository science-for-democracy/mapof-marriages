import pytest
import numpy as np

import mapof.marriages as mapel


registered_marriages_features_to_test = {
    'summed_rank_minimal_matching',
    'summed_rank_maximal_matching',
    'minimal_rank_maximizing_matching',
    'avg_num_of_bps_for_rand_matching',
    'num_of_bps_min_weight',
}

class TestFeatures:

    @pytest.mark.parametrize("feature_id", registered_marriages_features_to_test)
    def test_marriages_features(self, feature_id):

        num_agents = int(np.random.randint(5, 50) * 2)

        instance = mapel.generate_marriages_instance(culture_id='ic',
                                                     num_agents=num_agents)

        instance.compute_feature(feature_id)
