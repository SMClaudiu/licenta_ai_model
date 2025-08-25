from src.models.classification_model import TaskStatusClassifier
from src.models.regression_model import TaskCompletionRegressor
from src.training.trainer import TaskModelTrainer


class TaskModelFactory:
    @staticmethod
    def create_classifier(input_size, num_classes=3, hidden_sizes=None, dropout_rate=0.4):
        if hidden_sizes is None:
            hidden_sizes = [256, 128]
        return TaskStatusClassifier(input_size, hidden_sizes, num_classes, dropout_rate)

    @staticmethod
    def create_regressor(input_size, hidden_sizes=None, dropout_rate=0.25):
        if hidden_sizes is None:
            if input_size <= 50:
                hidden_sizes = [256, 128, 64]
            elif input_size <= 100:
                hidden_sizes = [512, 256, 128]
            else:
                hidden_sizes = [1024, 512, 256]
        return TaskCompletionRegressor(input_size, hidden_sizes, dropout_rate)

    @staticmethod
    def create_trainer(model, device='cpu', class_weights=None):
        return TaskModelTrainer(model, device, class_weights)