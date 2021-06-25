"""
TODO:
1. each chunk should be represented as a sequence of len/9 timepoints meaning each timepoint is a 9-mer
2. a vocabulary of 9-mers should be built meaning there should 4**9 vocab values (myb thinking about doing sth similar to BPE for "tokenization") - how to build initial embeddings? -> word2vec
3. a transformer network should be learning representations using triplet loss as an evaluation metric
4. the chunk representation should be learned using pooling on kmer embedding since it has been shown as better than using the CLS token - however, think about the CLS token as well
"""
import math
import torch
import torch.nn as nn
from sklearn.preprocessing import normalize


class RepresentationNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, attention_heads, dim_feedeforward, dropout, num_transformer_layers, device):
        super(RepresentationNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedder = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoder = PositionalEncoder(embedding_dim, device)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=attention_heads, dim_feedforward=dim_feedeforward, dropout=dropout)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_transformer_layers)
        self.batch_norm = nn.BatchNorm1d(num_features=embedding_dim)
        self.device = device

    def forward(self, x, original_lens, slice_indices):
        x = self.embedder(x) * math.sqrt(self.embedding_dim)
        x = self.positional_encoder(x)
        x = self.transformer(x)
        for i in range(len(original_lens)):
            if original_lens[i] < x.size()[1]:
                x[i, original_lens[i]:, :] = 0.0
            elif original_lens[i] > x.size()[1]:
                original_lens[i] = x.size()[1]
        x = torch.sum(x, dim=1)
        original_lens = torch.tensor(original_lens).unsqueeze(1).to(self.device)
        x = x / original_lens
        #x = torch.mean(x, 1)
        #print(x_1, x)
        #print(x_1.size(), x.size())
        x = torch.split(x, slice_indices, 0)
        x = torch.cat([torch.mean(i, 0, keepdim=True) for i in x], 0)
        x = self.batch_norm(x)
        return x

    def get_loss(self, anchors, positives, negatives, margin=2.5, alpha=0.9):
        #anchors = self.normalize(anchors)
        #positives = self.normalize(positives)
        #negatives = self.normalize(negatives)
        losses = torch.sum((anchors - positives) ** 2, dim=1) - torch.sum((anchors - negatives) ** 2, dim=1) + margin
        # if epoch % 50 == 0:
        #    print("First anchor: {}, positive: {} and negative: {}".format(anchors[0], positives[0], negatives[0]))
        #    print("Losses for pairs: {}".format(torch.transpose(losses, 0, 1)))
        zeros = torch.zeros(losses.size()).to(self.device)
        triplet_loss = alpha * torch.mean(torch.max(losses, zeros))
        return triplet_loss


class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim, device, max_len=5000, dropout=0.1):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe = pe.to(device)

    def forward(self, x):
        x = x + self.pe[:x.size()[0], :]
        return self.dropout(x)


