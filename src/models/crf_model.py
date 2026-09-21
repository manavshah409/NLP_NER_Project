"""Native CRFsuite model storage; no pickle deserialisation."""
import pycrfsuite


class CRFModel:
    def __init__(self, config):
        self.config = config
        self.tagger = None

    def trainer(self):
        trainer = pycrfsuite.Trainer(algorithm=self.config["algorithm"], verbose=True)
        trainer.set_params({"c1": self.config["c1"], "c2": self.config["c2"],
                            "max_iterations": self.config["max_iterations"],
                            "feature.possible_transitions": self.config["all_possible_transitions"]})
        return trainer

    def load(self, path):
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(str(path))
        return self

    def predict(self, features):
        if self.tagger is None:
            raise ValueError("Model not loaded")
        if not features:
            return []
        result = self.tagger.tag(features)
        if len(result) != len(features):
            raise ValueError("CRF output shape mismatch")
        return result
