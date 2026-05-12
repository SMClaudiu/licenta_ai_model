from torch import nn


class TaskCompletionRegressor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], dropout_rate=0.25):
        super(TaskCompletionRegressor, self).__init__()
        self.input_size = input_size
        self.input_bn = nn.BatchNorm1d(input_size)
        self.layers = nn.ModuleList()
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            current_dropout = dropout_rate * (1 - i * 0.05)
            current_dropout = max(current_dropout, 0.1)
            block = nn.Sequential(
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.GELU(),
                nn.Dropout(current_dropout)
            )
            self.layers.append(block)
            prev_size = hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(prev_size, prev_size // 2),
            nn.BatchNorm1d(prev_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(prev_size // 2, 1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_bn(x)
        for layer in self.layers:
            x = layer(x)
        output = self.regressor(x)
        return output.squeeze()