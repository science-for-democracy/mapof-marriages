
import mapof.marriages as mapof


class TestOfflineMarriagesExperiment:

    def setup_method(self):
        self.experiment = mapof.prepare_offline_marriages_experiment(experiment_id="test_id")

    def test_prepare_instances(self):
        self.experiment.prepare_instances()

    def test_compute_distances(self):
        self.experiment.prepare_instances()
        self.experiment.compute_distances(distance_id="l1-mutual_attraction")

    def test_embed_2d(self):
        self.experiment.prepare_instances()
        self.experiment.compute_distances(distance_id="l1-mutual_attraction")
        self.experiment.embed_2d(embedding_id="kk")

    def test_print_map_2d(self):
        self.experiment.prepare_instances()
        self.experiment.compute_distances(distance_id="l1-mutual_attraction")
        self.experiment.embed_2d(embedding_id="kk")
        self.experiment.print_map_2d(show=False)


