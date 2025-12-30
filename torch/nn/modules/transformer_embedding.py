from .sparse import Embedding
import torch 

class TransformerEmbedding(Embedding):
 def __init__(self, d_model, max_length, vocab):
  super().__init__()
  self.embedding_layer = Embedding(embedding_dim = d_model, num_embeddings = vocab)
  self.pos_embed = Embedding(embedding_dim = d_model, num_embeddings = max_length)

 def forward(self, x):
  batch, length = x.shape

  return self.embedding_layer(x) + self.pos_embed(torch.arange(0, length).expand(batch, length))
 
