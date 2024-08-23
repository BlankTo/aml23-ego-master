from torch import nn

class MLP(nn.Module):
    def __init__(self, num_classes, batch_size):
        super(MLP, self).__init__()
        self.input_size = 1024
        self.mlp = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        logits = self.mlp(x)
        return logits, {"features": {}}

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """

    def forward(self, x):
        return self.classifier(x), {}
