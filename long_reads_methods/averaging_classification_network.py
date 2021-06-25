from torch import nn


class ClassificationNetwork(nn.Module):
    def __init__(self, embedding_module, embedding_dim, classes, freeze_embeddings=True):
        super(ClassificationNetwork, self).__init__()
        self.embedding_module = embedding_module
        self.classification_head = nn.Linear(embedding_dim, classes)
        if freeze_embeddings:
            self.freeze_embeddings()

    def freeze_embeddings(self):
        print('Freezing embedding layers')
        for param in self.embedding_module.parameters():
            param.requires_grad = False

    def forward(self, x, lens, slices):
        x = self.embedding_module.forward(x, lens, slices)
        x = self.classification_head(x)
        return x
