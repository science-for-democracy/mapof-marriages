import mapof.marriages as mapof


class TestMultipleProcesses:

    def setup_method(self):
        self.experiment = mapof.prepare_offline_marriages_experiment(experiment_id="test_id")

    def test_multiple_processes(self):
        self.experiment.prepare_instances()
        self.experiment.compute_distances(distance_id="l1-mutual_attraction", num_processes=2)