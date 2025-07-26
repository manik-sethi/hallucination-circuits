#src/detectors/facs_score.py
import requests
import json
from src.interfaces.neuronpedia_api import NeuronpediaClient

class FACSScore():
    def __init__(self, query, model_id, sae_layer, source_set):
        self.query = query
        self.model_id = model_id
        self.sae_layer = sae_layer
        self.source_set = source_set


        self.F_expected = []
        self.F_actual = []

    def compute_F_expected(self):
        print("let's print out the all_text_feat output")
        npedia = NeuronpediaClient(
            model_id=self.model_id,
            sae_layer=self.sae_layer,
            source_set=self.source_set
        )

        top_feats = npedia.all_text_feat(
            query=self.query,
            ignore_bos=True,
            density_threshold=0.1,
            num_results=15
        )
        return top_feats


# npedia = FACSScore(**param)
