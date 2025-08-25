

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, r2_score, mean_absolute_error
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from tqdm import tqdm
import math


class TaskModelTrainer:
    def __init__(self, model, device='cpu', class_weights=None):
        self.model = model
        self.device = device
        self.class_weights = class_weights
        self.model.to(device)
        self.train_losses, self.val_losses, self.train_metrics, self.val_metrics, self.learning_rates = [], [], [], [], []
        self.best_val_metric = 0.0 if hasattr(model, 'num_classes') else float('inf')
        self.patience_counter = 0
        self.training_stopped_early = False

    def train_classification(self, train_loader, val_loader, epochs=200, lr=0.001, weight_decay=1e-4):
        if self.class_weights is None:
            all_targets = np.concatenate([targets.numpy() for _, targets in train_loader])
            class_weights = compute_class_weight('balanced', classes=np.unique(all_targets), y=all_targets)
            class_weights = torch.FloatTensor(class_weights).to(self.device)
        else:
            class_weights = torch.FloatTensor(self.class_weights).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs, pct_start=0.3)
        early_stop_patience = 50
        min_delta = 0.0005
        best_val_acc = 0.0
        for epoch in tqdm(range(epochs), desc="Training Classification"):
            self.model.train()
            train_loss, train_predictions, train_targets = 0.0, [], []
            for batch_features, batch_targets in train_loader:
                batch_features, batch_targets = batch_features.to(self.device), batch_targets.to(self.device).long()
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                optimizer.step()
                train_loss += loss.item()
                train_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                train_targets.extend(batch_targets.cpu().numpy())
                scheduler.step()
            val_loss, val_predictions, val_targets = self._validate(val_loader, criterion)
            train_acc = accuracy_score(train_targets, train_predictions)
            val_acc = accuracy_score(val_targets, val_predictions)
            self.train_losses.append(train_loss / len(train_loader))
            self.val_losses.append(val_loss / len(val_loader))
            self.train_metrics.append(train_acc)
            self.val_metrics.append(val_acc)
            self.learning_rates.append(optimizer.param_groups[0]['lr'])
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'artifacts/classification_model.pth')
            else:
                self.patience_counter += 1
            if self.patience_counter >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch + 1} as validation accuracy did not improve by more than {min_delta} for {early_stop_patience} epochs.")
                self.training_stopped_early = True
                break
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {self.train_losses[-1]:.4f}, Val Loss: {self.val_losses[-1]:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")

    def train_regression(self, train_loader, val_loader, epochs=200, lr=0.001, weight_decay=1e-4):
        def combined_loss(pred, target):
            return 0.7 * nn.MSELoss()(pred, target) + 0.3 * nn.L1Loss()(pred, target)
        criterion = combined_loss
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=20, min_lr=1e-6, verbose=True)
        best_val_loss = float('inf')
        early_stop_patience = 60
        min_delta = 0.001
        for epoch in tqdm(range(epochs), desc="Training Regression"):
            self.model.train()
            train_loss, train_predictions, train_targets = 0.0, [], []
            for batch_features, batch_targets in train_loader:
                batch_features, batch_targets = batch_features.to(self.device), batch_targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                optimizer.step()
                train_loss += loss.item()
                train_predictions.extend(outputs.detach().cpu().numpy())
                train_targets.extend(batch_targets.cpu().numpy())
            val_loss, val_predictions, val_targets = self._validate_regression(val_loader, criterion)
            train_r2 = r2_score(train_targets, train_predictions)
            val_r2 = r2_score(val_targets, val_predictions)
            self.train_losses.append(train_loss / len(train_loader))
            self.val_losses.append(val_loss / len(val_loader))
            self.train_metrics.append(train_r2)
            self.val_metrics.append(val_r2)
            self.learning_rates.append(optimizer.param_groups[0]['lr'])
            scheduler.step(self.val_losses[-1])
            if self.val_losses[-1] < best_val_loss - min_delta:
                best_val_loss = self.val_losses[-1]
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'artifacts/regression_model.pth')
            else:
                self.patience_counter += 1
            if self.patience_counter >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch + 1} as validation loss did not improve by more than {min_delta} for {early_stop_patience} epochs.")
                self.training_stopped_early = True
                break
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {self.train_losses[-1]:.4f}, Val Loss: {self.val_losses[-1]:.4f}, Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}")
        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")

    def _validate(self, val_loader, criterion):
        self.model.eval()
        val_loss, val_predictions, val_targets = 0.0, [], []
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features, batch_targets = batch_features.to(self.device), batch_targets.to(self.device).long()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
                val_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_targets.extend(batch_targets.cpu().numpy())
        return val_loss, val_predictions, val_targets

    def _validate_regression(self, val_loader, criterion):
        self.model.eval()
        val_loss, val_predictions, val_targets = 0.0, [], []
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features, batch_targets = batch_features.to(self.device), batch_targets.to(self.device)
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(batch_targets.cpu().numpy())
        return val_loss, val_predictions, val_targets

    def evaluate_classification(self, test_loader):
        self.model.eval()
        test_predictions, test_targets, test_probabilities = [], [], []
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features, batch_targets = batch_features.to(self.device), batch_targets.to(self.device).long()
                outputs = self.model(batch_features)
                probabilities = torch.softmax(outputs, dim=1)
                test_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                test_targets.extend(batch_targets.cpu().numpy())
                test_probabilities.extend(probabilities.cpu().numpy())
        accuracy = accuracy_score(test_targets, test_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(test_targets, test_predictions, average='macro')
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(test_targets, test_predictions, average=None)
        print("\n🎯 Classification Test Results:")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Macro Precision: {precision:.4f}")
        print(f"Macro Recall: {recall:.4f}")
        print(f"Macro F1-Score: {f1:.4f}")
        print("\nPer-class Performance:")
        class_names = ['Pending', 'In Progress', 'Completed']
        for i, class_name in enumerate(class_names):
            if i < len(precision_per_class):
                print(f"{class_name}: Precision={precision_per_class[i]:.4f}, Recall={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}")
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'per_class_precision': precision_per_class, 'per_class_recall': recall_per_class, 'per_class_f1': f1_per_class, 'predictions': test_predictions, 'targets': test_targets, 'probabilities': test_probabilities, 'stopped_early': self.training_stopped_early}

    def evaluate_regression(self, test_loader):
        self.model.eval()
        test_predictions, test_targets = [], []
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features, batch_targets = batch_features.to(self.device), batch_targets.to(self.device)
                outputs = self.model(batch_features)
                test_predictions.extend(outputs.cpu().numpy())
                test_targets.extend(batch_targets.cpu().numpy())
        test_predictions = np.array(test_predictions)
        test_targets = np.array(test_targets)
        mse = mean_squared_error(test_targets, test_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_targets, test_predictions)
        r2 = r2_score(test_targets, test_predictions)
        mape = np.mean(np.abs((test_targets - test_predictions) / (test_targets + 1e-8))) * 100
        print("\n📈 Regression Test Results:")
        print(f"RMSE: {rmse:.4f} days")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f} days")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        return {'rmse': rmse, 'mse': mse, 'mae': mae, 'r2': r2, 'mape': mape, 'predictions': test_predictions, 'targets': test_targets, 'stopped_early': self.training_stopped_early}

    def plot_training_history(self):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        if len(self.train_losses) > 10:
            window = min(10, len(self.train_losses) // 20 + 1)
            train_smooth = pd.Series(self.train_losses).rolling(window=window, center=True).mean()
            val_smooth = pd.Series(self.val_losses).rolling(window=window, center=True).mean()
            axes[0, 0].plot(self.train_losses, alpha=0.3, color='blue', label='Training Loss (raw)')
            axes[0, 0].plot(train_smooth, color='blue', linewidth=2, label='Training Loss (smooth)')
            axes[0, 0].plot(self.val_losses, alpha=0.3, color='orange', label='Validation Loss (raw)')
            axes[0, 0].plot(val_smooth, color='orange', linewidth=2, label='Validation Loss (smooth)')
        else:
            axes[0, 0].plot(self.train_losses, label='Training Loss')
            axes[0, 0].plot(self.val_losses, label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        if self.training_stopped_early:
            axes[0, 0].axvline(x=len(self.train_losses) - 1, color='red', linestyle='--', label=f'Early Stop (epoch {len(self.train_losses)})')
            axes[0, 0].legend()
        if len(self.train_metrics) > 0:
            axes[0, 1].plot(self.train_metrics, label='Training Metric', linewidth=2)
            axes[0, 1].plot(self.val_metrics, label='Validation Metric', linewidth=2)
            axes[0, 1].set_title('Training and Validation Metrics', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Metric')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        if len(self.learning_rates) > 0:
            axes[1, 0].plot(self.learning_rates, color='red', linewidth=2, label='Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        if len(self.train_losses) > 0 and len(self.val_losses) > 0:
            generalization_gap = [abs(t - v) for t, v in zip(self.train_losses, self.val_losses)]
            axes[1, 1].plot(generalization_gap, color='purple', linewidth=2, label='Generalization Gap')
            axes[1, 1].set_title('Generalization Gap Analysis', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('|Train Loss - Val Loss|')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            if len(generalization_gap) > 10:
                z = np.polyfit(range(len(generalization_gap)), generalization_gap, 1)
                p = np.poly1d(z)
                axes[1, 1].plot(range(len(generalization_gap)), p(range(len(generalization_gap))), "--", alpha=0.8, color='red', label='Trend')
                axes[1, 1].legend()
        plt.tight_layout()
        plt.suptitle('Training Analysis Dashboard', fontsize=16, fontweight='bold', y=1.02)
        plt.show()

    def predict(self, features, return_confidence=False):
        self.model.eval()
        with torch.no_grad():
            features = torch.FloatTensor(features).to(self.device)
            if len(features.shape) == 1:
                features = features.unsqueeze(0)
            outputs = self.model(features)
            if hasattr(self.model, 'num_classes'):
                predictions = torch.argmax(outputs, dim=1)
                probabilities = torch.softmax(outputs, dim=1)
                if return_confidence:
                    max_probs = torch.max(probabilities, dim=1)[0]
                    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)
                    normalized_entropy = entropy / math.log(probabilities.shape[1])
                    confidence = max_probs * (1 - normalized_entropy)
                    return (predictions.cpu().numpy(), probabilities.cpu().numpy(), confidence.cpu().numpy())
                else:
                    return predictions.cpu().numpy(), probabilities.cpu().numpy()
            else:
                numpy_outputs = outputs.cpu().numpy()
                return np.atleast_1d(numpy_outputs)