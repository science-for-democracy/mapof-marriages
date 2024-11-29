import mapof.marriages.features.basic_features as exp_mar

from mapof.marriages.features.register import registered_marriages_distances

def get_feature(feature_id):
        return registered_marriages_distances.get(feature_id)
