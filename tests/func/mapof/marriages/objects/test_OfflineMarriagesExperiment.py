import pytest

import mapof.marriages as mapof

@pytest.fixture(autouse=True)
def mock_path(mocker, tmp_path):
    mocker.patch("os.getcwd", return_value=str(tmp_path))
    mocker.patch("pathlib.Path.cwd", return_value=tmp_path)

@pytest.fixture
def offline_experiment():
    return mapof.prepare_offline_marriages_experiment(experiment_id="test_id")


@pytest.fixture
def prepared_instances(offline_experiment):
    offline_experiment.prepare_instances()
    return offline_experiment


class TestOfflineMarriagesExperiment:

    def test_compute_distances(self, prepared_instances):
        prepared_instances.compute_distances(distance_id="l1-mutual_attraction")

    # def test_embed_2d(self, prepared_instances):
    #     prepared_instances.compute_distances(distance_id="l1-mutual_attraction")
    #     prepared_instances.embed_2d(embedding_id="kk")
    #
    # def test_print_map_2d(self, prepared_instances):
    #     prepared_instances.compute_distances(distance_id="l1-mutual_attraction")
    #     prepared_instances.embed_2d(embedding_id="kk")
    #     prepared_instances.print_map_2d(show=False)

    def test_compute_feature(self, prepared_instances):
        feature_id = 'summed_rank_minimal_matching'
        prepared_instances.compute_feature(feature_id=feature_id)



