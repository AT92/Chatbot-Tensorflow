import json


class ModelConfiguration:
	"""
		This class represants the configuration of the models.

		It represants the configuration for the word-embedding model as well as for the
		classification model.

		Attributes:
			parameters: a json object containing the read configuration from ./clasifier/configuration.json
	"""

	def __init__(self):
		"""Inits the ModelConfiguration object, by reading the configuration.json file."""
		self.parameters = json.load(open("./classifier/configuration.json"))

	def get_word_embedding_config(self):
		"""returns the configuration for the word embedding model"""
		return self.parameters["networks"][0]

	def get_classifier_config(self):
		"""returns the configuration for the question classification model"""
		return self.parameters["networks"][1]

	def save(self):
		"""saves the updated configuration to the ./classifier/configuration.json file"""
		with open('./classifier/configuration.json', 'w') as outfile:
			json.dump(self.parameters, outfile, indent=4, sort_keys=True)
