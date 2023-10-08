from analysis.config import Config, EmbeddingConfig, ModelSelectionConfig
from geometric_encoding.train import triplet_train

from geometric_encoding.embedding import LinearEmbedder

config = Config.from_default()

model = LinearEmbedder(input_size=None, output_size=config.embedding.model.dim, dropout=config.embedding.model.dropout)

model_kws = config.embedding.model.to_dict()
train_kws = config.embedding.training.to_dict()
