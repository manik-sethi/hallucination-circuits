#src/interfaces/neuronpedia_api.py
import requests
import json


class NeuronpediaClient():
    def __init__(self, model_id, sae_layer, source_set):
        self.model_id = model_id
        self.sae_layer = sae_layer
        self.source_set = source_set


    def feat_specific_act(self, index, text):
            url = "https://www.neuronpedia.org/api/activation/new"

            payload = {
                "feature": {
                    "modelId": self.model_id,
                    "source": self.sae_layer,
                    "index": index
                },
                "customText": text
                }

            headers = {
                "Content-Type": "applications/json"
            }

            try:
                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error making request: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Response text: {e.response.text}")
                return None


    def all_text_feat(self, query, ignore_bos, density_threshold, num_results):

            url = "https://www.neuronpedia.org/api/search-all"

            payload = {
                "modelId": self.model_id,
                "sourceSet": self.source_set,  # This is the SAE identifier (layer-type-source)
                "text": query,
                "selectedLayers": [
                    self.sae_layer
                ],
                "sortIndexes": [
                    1
                ],
                "ignoreBos": ignore_bos,
                "densityThreshold": density_threshold,
                "numResults": num_results
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            try:
                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error making request: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Response text: {e.response.text}")
                return None


