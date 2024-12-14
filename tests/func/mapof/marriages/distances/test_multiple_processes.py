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

class TestMultipleProcesses:

    def test_multiple_processes(self, prepared_instances):
        prepared_instances.compute_distances(distance_id="l1-mutual_attraction", num_processes=2)
