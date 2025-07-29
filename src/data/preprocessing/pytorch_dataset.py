"""
Enhanced Machine Learning Models for Task Management
Key improvements for better AI performance
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, r2_score, \
    mean_absolute_error
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


class TaskStatusClassifier(nn.Module):

    def __init__(self, input_size, hidden_sizes=[256, 128, 64], num_classes=3, dropout_rate=0.4):
        super(TaskStatusClassifier, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        #Input normalization
        self.input_bn = nn.BatchNorm1d(input_size)

        #Build deeper network with residual connections
        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            #Main path
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

            #Add residual connection if dimensions match
            if prev_size == hidden_size and i > 0:
                self.add_residual = True

            prev_size = hidden_size

        self.feature_layers = nn.Sequential(*layers)

        #Output layers with different dropout
        self.classifier = nn.Sequential(
            nn.Linear(prev_size, prev_size // 2),
            nn.BatchNorm1d(prev_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),  # Lower dropout before output
            nn.Linear(prev_size // 2, num_classes)
        )

        #Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):

        #Initialize weights using Xavier/He initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input normalization
        x = self.input_bn(x)

        # Feature extraction
        features = self.feature_layers(x)

        # Classification
        output = self.classifier(features)

        return output


class TaskCompletionRegressor(nn.Module):

    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout_rate=0.3):
        super(TaskCompletionRegressor, self).__init__()

        self.input_size = input_size

        #Input normalization
        self.input_bn = nn.BatchNorm1d(input_size)

        #Build network with skip connections
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ELU())  # ELU can work better than ReLU for regression
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        self.feature_layers = nn.Sequential(*layers)

        #Output layer with different activation
        self.regressor = nn.Sequential(
            nn.Linear(prev_size, prev_size // 2),
            nn.BatchNorm1d(prev_size // 2),
            nn.ELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(prev_size // 2, 1),
            nn.Softplus()  # Ensures positive output for time prediction
        )

        #Initialize weights
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
        #Input normalization
        x = self.input_bn(x)

        #Feature extraction
        features = self.feature_layers(x)

        #Regression output
        output = self.regressor(features)

        return output.squeeze()


class TaskModelTrainer:

    def __init__(self, model, device='cpu', class_weights=None):
        self.model = model
        self.device = device
        self.class_weights = class_weights
        self.model.to(device)
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.learning_rates = []

    def train_classification(self, train_loader, val_loader, epochs=150, lr=0.001, weight_decay=1e-4):

        # Calculate class weights if not provided
        if self.class_weights is None:
            # Extract all targets to compute class weights
            all_targets = []
            for _, targets in train_loader:
                all_targets.extend(targets.numpy())

            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(all_targets),
                y=all_targets
            )
            class_weights = torch.FloatTensor(class_weights).to(self.device)
        else:
            class_weights = torch.FloatTensor(self.class_weights).to(self.device)

        # Use weighted loss
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        #Optimizer with different learning rates for different layers
        optimizer = optim.AdamW([
            {'params': self.model.feature_layers.parameters(), 'lr': lr},
            {'params': self.model.classifier.parameters(), 'lr': lr * 0.1}  # Lower LR for classifier
        ], weight_decay=weight_decay)

        #Scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

        best_val_acc = 0.0
        patience_counter = 0
        early_stop_patience = 30

        #Data augmentation through mixup (optional)
        mixup_alpha = 0.2

        for epoch in tqdm(range(epochs), desc="Training Classification"):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_predictions = []
            train_targets = []

            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device).long()

                #Mixup augmentation
                if np.random.random() > 0.7:  # Apply mixup 30% of the time
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                    index = torch.randperm(batch_features.size(0)).to(self.device)
                    mixed_features = lam * batch_features + (1 - lam) * batch_features[index, :]
                    targets_a, targets_b = batch_targets, batch_targets[index]

                    optimizer.zero_grad()
                    outputs = self.model(mixed_features)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                    loss.backward()

                    #Gradient
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                    train_loss += loss.item()
                    train_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    train_targets.extend(targets_a.cpu().numpy())
                else:
                    optimizer.zero_grad()
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                    train_loss += loss.item()
                    train_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    train_targets.extend(batch_targets.cpu().numpy())

            #Validation phase
            self.model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device).long()

                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_targets)

                    val_loss += loss.item()
                    val_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    val_targets.extend(batch_targets.cpu().numpy())

            train_acc = accuracy_score(train_targets, train_predictions)
            val_acc = accuracy_score(val_targets, val_predictions)

            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metrics.append(train_acc)
            self.val_metrics.append(val_acc)
            self.learning_rates.append(optimizer.param_groups[0]['lr'])

            scheduler.step()

            #Early stopping with improved criteria
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_classification_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if epoch % 20 == 0:
                print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")

    def train_regression(self, train_loader, val_loader, epochs=150, lr=0.001, weight_decay=1e-4):

        #Use Huber loss which is more robust to outliers
        criterion = nn.HuberLoss(delta=1.0)

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.001)

        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 30

        for epoch in tqdm(range(epochs), desc="Training Regression"):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_predictions = []
            train_targets = []

            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                train_predictions.extend(outputs.detach().cpu().numpy())
                train_targets.extend(batch_targets.cpu().numpy())

            #Validation phase
            self.model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)

                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_targets)

                    val_loss += loss.item()
                    val_predictions.extend(outputs.cpu().numpy())
                    val_targets.extend(batch_targets.cpu().numpy())

            #Calculate metrics
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
            self.learning_rates.append(optimizer.param_groups[0]['lr'])

            scheduler.step()

            #Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_regression_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if epoch % 20 == 0:
                print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}, "
                      f"Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}")

        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")

    def evaluate_classification(self, test_loader):
        self.model.eval()
        test_predictions = []
        test_targets = []
        test_probabilities = []

        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device).long()

                outputs = self.model(batch_features)
                probabilities = torch.softmax(outputs, dim=1)

                test_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                test_targets.extend(batch_targets.cpu().numpy())
                test_probabilities.extend(probabilities.cpu().numpy())

        #Calculate metrics
        accuracy = accuracy_score(test_targets, test_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(test_targets, test_predictions, average='macro')

        #Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            test_targets, test_predictions, average=None
        )

        print(f"Test Results:")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Macro Precision: {precision:.4f}")
        print(f"Macro Recall: {recall:.4f}")
        print(f"Macro F1-Score: {f1:.4f}")

        print(f"\nPer-class Performance:")
        class_names = ['Pending', 'In Progress', 'Completed']
        for i, class_name in enumerate(class_names):
            print(f"{class_name}: Precision={precision_per_class[i]:.4f}, "
                  f"Recall={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'per_class_precision': precision_per_class,
            'per_class_recall': recall_per_class,
            'per_class_f1': f1_per_class,
            'predictions': test_predictions,
            'targets': test_targets,
            'probabilities': test_probabilities
        }

    def evaluate_regression(self, test_loader):
        self.model.eval()
        test_predictions = []
        test_targets = []
        test_loss = 0.0
        criterion = nn.HuberLoss(delta=1.0)

        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_targets)
                test_loss += loss.item()

                test_predictions.extend(outputs.cpu().numpy())
                test_targets.extend(batch_targets.cpu().numpy())

        #Calculate comprehensive regression metrics
        mse = mean_squared_error(test_targets, test_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_targets, test_predictions)
        r2 = r2_score(test_targets, test_predictions)

        print(f"\nEnhanced Regression Test Results:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")

        return {
            'rmse': rmse,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'predictions': test_predictions,
            'targets': test_targets,
            'loss': test_loss / len(test_loader)
        }

    def plot_enhanced_training_history(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        #Loss plot
        axes[0, 0].plot(self.train_losses, label='Training Loss', alpha=0.8)
        axes[0, 0].plot(self.val_losses, label='Validation Loss', alpha=0.8)
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        #Metrics plot
        if len(self.train_metrics) > 0:
            axes[0, 1].plot(self.train_metrics, label='Training Metric', alpha=0.8)
            axes[0, 1].plot(self.val_metrics, label='Validation Metric', alpha=0.8)
            axes[0, 1].set_title('Training and Validation Metrics')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Metric')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        #Learning rate plot
        if len(self.learning_rates) > 0:
            axes[1, 0].plot(self.learning_rates, label='Learning Rate', color='red', alpha=0.8)
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        #Loss difference plot
        if len(self.train_losses) > 0 and len(self.val_losses) > 0:
            loss_diff = [abs(t - v) for t, v in zip(self.train_losses, self.val_losses)]
            axes[1, 1].plot(loss_diff, label='|Train Loss - Val Loss|', color='purple', alpha=0.8)
            axes[1, 1].set_title('Training-Validation Loss Difference')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Absolute Difference')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def predict(self, features, return_confidence=False):
        #Confidence scores prediction
        self.model.eval()
        with torch.no_grad():
            features = torch.FloatTensor(features).to(self.device)
            if len(features.shape) == 1:
                features = features.unsqueeze(0)

            outputs = self.model(features)

            if hasattr(self.model, 'num_classes'):
                #Classification
                predictions = torch.argmax(outputs, dim=1)
                probabilities = torch.softmax(outputs, dim=1)

                if return_confidence:
                    #Calculate confidence as max probability
                    confidence = torch.max(probabilities, dim=1)[0]
                    return predictions.cpu().numpy(), probabilities.cpu().numpy(), confidence.cpu().numpy()
                else:
                    return predictions.cpu().numpy(), probabilities.cpu().numpy()
            else:
                # Regression
                return outputs.cpu().numpy()



class TaskModelFactory:

    @staticmethod
    def create_classifier(input_size, num_classes=3, hidden_sizes=[256, 128, 64], dropout_rate=0.4):
        """Create the task status classifier"""
        return TaskStatusClassifier(input_size, hidden_sizes, num_classes, dropout_rate)

    @staticmethod
    def create_regressor(input_size, hidden_sizes=[256, 128, 64], dropout_rate=0.3):
        """Create the completion time regressor"""
        return TaskCompletionRegressor(input_size, hidden_sizes, dropout_rate)

    @staticmethod
    def create_trainer(model, device='cpu', class_weights=None):
        """Create the model trainer"""
        return TaskModelTrainer(model, device, class_weights)

