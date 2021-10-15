from torch import nn


class Feedforward(nn.Module):
    '''Neural Network's architecture'''

    def __init__(self, input_size, hidden_size, num_classes):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # 2 hidden layers, ReLU and Dropout
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.hidden_size, num_classes)
        )

    # forward pass
    def forward(self, x):
        return self.model(x)
