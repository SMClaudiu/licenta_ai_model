import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm


class TaskStatusClassifier(nn.Module):
    """
    Neural network for predicting task status (Pending, In Progress, Completed)
    """

    def __init__(self, input_size, hidden_sizes=[64, 32], num_classes=3, dropout_rate=0.3):
        super(TaskStatusClassifier, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        # Build layers dynamically
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TaskCompletionRegressor(nn.Module):
    """
    Neural network for predicting task completion time in days
    """

    def __init__(self, input_size, hidden_sizes=[64, 32], dropout_rate=0.3):
        super(TaskCompletionRegressor, self).__init__()

        self.input_size = input_size

        # Build layers dynamically
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer (single value for regression)
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()


class TaskModelTrainer:
    """
    Trainer class for both classification and regression models
    """

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []

    def train_classification(self, train_loader, val_loader, epochs=100, lr=0.001, weight_decay=1e-5):
        """Train classification model"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10)

        best_val_acc = 0.0
        patience_counter = 0
        early_stop_patience = 20

        for epoch in tqdm(range(epochs), desc="Training"):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_predictions = []
            train_targets = []

            for batch_features, batch_targets in train_loader:
                batch_features, batch_targets = batch_features.to(self.device), batch_targets.to(self.device)
                batch_targets = batch_targets.long()

                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                train_targets.extend(batch_targets.cpu().numpy())

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features, batch_targets = batch_features.to(self.device), batch_targets.to(self.device)
                    batch_targets = batch_targets.long()

                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_targets)

                    val_loss += loss.item()
                    val_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    val_targets.extend(batch_targets.cpu().numpy())

            # Calculate metrics
            train_acc = accuracy_score(train_targets, train_predictions)
            val_acc = accuracy_score(val_targets, val_predictions)

            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metrics.append(train_acc)
            self.val_metrics.append(val_acc)

            scheduler.step(val_loss)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_classification_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")

    def train_regression(self, train_loader, val_loader, epochs=100, lr=0.001, weight_decay=1e-5):
        """Train regression model"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10)

        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 20

        for epoch in tqdm(range(epochs), desc="Training"):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_predictions = []
            train_targets = []

            for batch_features, batch_targets in train_loader:
                batch_features, batch_targets = batch_features.to(self.device), batch_targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_predictions.extend(outputs.detach().cpu().numpy())
                train_targets.extend(batch_targets.cpu().numpy())

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features, batch_targets = batch_features.to(self.device), batch_targets.to(self.device)

                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_targets)

                    val_loss += loss.item()
                    val_predictions.extend(outputs.cpu().numpy())
                    val_targets.extend(batch_targets.cpu().numpy())

            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(train_targets, train_predictions))
            val_rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
            train_r2 = r2_score(train_targets, train_predictions)
            val_r2 = r2_score(val_targets, val_predictions)

            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metrics.append(train_r2)
            self.val_metrics.append(val_r2)

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_regression_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}, "
                      f"Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}")

        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")

    def evaluate_classification(self, test_loader):
        """Evaluate classification model on test set"""
        self.model.eval()
        test_predictions = []
        test_targets = []

        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features, batch_targets = batch_features.to(self.device), batch_targets.to(self.device)
                batch_targets = batch_targets.long()

                outputs = self.model(batch_features)
                test_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                test_targets.extend(batch_targets.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(test_targets, test_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(test_targets, test_predictions, average='macro')

        print(f"Test Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': test_predictions,
            'targets': test_targets
        }

    def evaluate_regression(self, test_loader):
        """Evaluate regression model on test set"""
        self.model.eval()
        test_predictions = []
        test_targets = []

        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features, batch_targets = batch_features.to(self.device), batch_targets.to(self.device)

                outputs = self.model(batch_features)
                test_predictions.extend(outputs.cpu().numpy())
                test_targets.extend(batch_targets.cpu().numpy())

        # Calculate metrics
        mse = mean_squared_error(test_targets, test_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(test_targets, test_predictions)
        mae = np.mean(np.abs(np.array(test_targets) - np.array(test_predictions)))

        print(f"Test Results:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"MAE: {mae:.4f}")

        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'predictions': test_predictions,
            'targets': test_targets
        }

    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Metrics plot
        if len(self.train_metrics) > 0:
            ax2.plot(self.train_metrics, label='Training Metric')
            ax2.plot(self.val_metrics, label='Validation Metric')
            ax2.set_title('Training and Validation Metrics')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Metric')
            ax2.legend()
            ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def predict(self, features):
        """Make predictions on new data"""
        self.model.eval()
        with torch.no_grad():
            features = torch.FloatTensor(features).to(self.device)
            if len(features.shape) == 1:
                features = features.unsqueeze(0)

            outputs = self.model(features)

            if hasattr(self.model, 'num_classes'):
                # Classification
                predictions = torch.argmax(outputs, dim=1)
                probabilities = torch.softmax(outputs, dim=1)
                return predictions.cpu().numpy(), probabilities.cpu().numpy()
            else:
                # Regression
                return outputs.cpu().numpy()


class TaskModelFactory:
    """Factory class to create and configure models"""

    @staticmethod
    def create_classifier(input_size, num_classes=3, hidden_sizes=[64, 32], dropout_rate=0.3):
        """Create a task status classifier"""
        return TaskStatusClassifier(input_size, hidden_sizes, num_classes, dropout_rate)

    @staticmethod
    def create_regressor(input_size, hidden_sizes=[64, 32], dropout_rate=0.3):
        """Create a task completion time regressor"""
        return TaskCompletionRegressor(input_size, hidden_sizes, dropout_rate)

    @staticmethod
    def create_trainer(model, device='cpu'):
        """Create a model trainer"""
        return TaskModelTrainer(model, device)