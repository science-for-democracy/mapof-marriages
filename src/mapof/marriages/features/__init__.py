import mapof.marriages.features.basic_features as exp_mar


def get_feature(feature_id):
    return {'summed_rank_maximal_matching': exp_mar.summed_rank_maximal_matching,
            'summed_rank_minimal_matching': exp_mar.summed_rank_minimal_matching,
            'minimal_rank_maximizing_matching': exp_mar.minimal_rank_maximizing_matching,
            'avg_num_of_bps_for_rand_matching': exp_mar.avg_number_of_bps_for_random_matching,
            'num_of_bps_min_weight': exp_mar.number_of_bps_maximumWeight,
            }.get(feature_id)
