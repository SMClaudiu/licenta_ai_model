import torch
import torch.nn as nn


class TaskStatusClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes=None, num_classes=3, dropout_rate=0.4):
        super(TaskStatusClassifier, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128]
        self.input_size = input_size
        self.num_classes = num_classes
        self.input_bn = nn.BatchNorm1d(input_size)
        self.layers = nn.ModuleList()
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            current_dropout = dropout_rate * (1 - i * 0.1)
            current_dropout = max(current_dropout, 0.1)
            block = nn.Sequential(
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(current_dropout)
            )
            self.layers.append(block)
            if prev_size == hidden_size:
                self.layers.append(ResidualBlock(hidden_size, current_dropout))
            prev_size = hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(prev_size, prev_size // 2),
            nn.BatchNorm1d(prev_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(prev_size // 2, num_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_bn(x)
        for layer in self.layers:
            if isinstance(layer, ResidualBlock):
                x = layer(x) + x
            else:
                x = layer(x)
        output = self.classifier(x)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, size, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(size, size),
            nn.BatchNorm1d(size)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x))

    def predict_proba(self , x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)