import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from src.core.config import AppConfig
from src.data.preprocessing import DataPreprocessor
from src.data.dataset import DatasetFactory
from src.models.factory import TaskModelFactory


def run_training_pipeline():
    config = AppConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- CLASSIFICATION ---
    print("\n" + "=" * 50)
    print("TASK STATUS CLASSIFICATION")
    print("=" * 50)

    classification_processor = DataPreprocessor()
    classification_dataset_factory = DatasetFactory(classification_processor)

    classification_data = classification_dataset_factory.create_classification_datasets(
         test_size=0.2 , val_size=0.15 , batch_size=config.model.batch_size
    )
    if classification_data is None:
        raise ValueError("Failed to create classification datasets.")

    classifier = TaskModelFactory.create_classifier(
        input_size=classification_data['num_features'] ,
        num_classes=classification_data['num_classes']
    )
    class_dist = classification_data['class_distribution']
    class_weights = [sum(class_dist.values()) / (len(class_dist) * count) for count in class_dist.values()]
    print(f"Using class weights: {class_weights}")

    classifier_trainer = TaskModelFactory.create_trainer(classifier , device , class_weights=class_weights)
    classifier_trainer.train_classification(
        train_loader=classification_data['train_loader'] ,
        val_loader=classification_data['val_loader'] ,
        epochs=config.model.max_epochs
    )
    classification_results = classifier_trainer.evaluate_classification(classification_data['test_loader'])

    classifier_trainer.plot_training_history()

    # --- REGRESSION ---
    print("\n" + "=" * 50)
    print("TASK COMPLETION TIME REGRESSION")
    print("=" * 50)

    regression_processor = DataPreprocessor()
    regression_dataset_factory = DatasetFactory(regression_processor)

    regression_data = regression_dataset_factory.create_regression_datasets(
       test_size=0.2 , val_size=0.15 , batch_size=config.model.batch_size
    )
    if regression_data is None:
        raise ValueError("Failed to create regression datasets.")

    regressor = TaskModelFactory.create_regressor(input_size=regression_data['num_features'])
    regressor_trainer = TaskModelFactory.create_trainer(regressor , device)
    regressor_trainer.train_regression(
        train_loader=regression_data['train_loader'] ,
        val_loader=regression_data['val_loader'] ,
        epochs=config.model.max_epochs
    )
    regression_results = regressor_trainer.evaluate_regression(regression_data['test_loader'])

    regressor_trainer.plot_training_history()

    print("\n" + "=" * 50)
    print("SAVING MODELS AND PREPROCESSORS")
    print("=" * 50)
    os.makedirs('artifacts' , exist_ok=True)

    classification_processor.save('artifacts/classification_preprocessors.pkl')
    regression_processor.save('artifacts/regression_preprocessors.pkl')

    torch.save({
                   'model_state_dict':classifier.state_dict() , 'model_config':{
            'input_size':classification_data['num_features'] , 'num_classes':classification_data['num_classes'] } } ,
               'artifacts/classification_model.pth')
    torch.save({
                   'model_state_dict':regressor.state_dict() ,
                   'model_config':{ 'input_size':regression_data['num_features'] } } , 'artifacts/regression_model.pth')
    print("Models and dedicated preprocessors saved successfully!")

    # --- PERFORMANCE SUMMARY ---
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    print("Classification Results:")
    print(f"  Overall Accuracy: {classification_results['accuracy']:.4f}")
    print(f"  Macro F1-Score: {classification_results['f1']:.4f}")
    create_confusion_matrix_plot(classification_results['targets'] , classification_results['predictions'])
    print(f"\nRegression Results:")
    print(f"  RMSE: {regression_results['rmse']:.4f} days")
    print(f"  R²: {regression_results['r2']:.4f}")


def create_confusion_matrix_plot(y_true , y_pred):
    plt.figure(figsize=(8 , 6))
    cm = confusion_matrix(y_true , y_pred)
    cm_normalized = cm.astype('float') / np.maximum(cm.sum(axis=1)[: , np.newaxis] , 1e-8)
    sns.heatmap(cm_normalized , annot=True , fmt='.3f' , cmap='Blues' ,
                xticklabels=['Pending' , 'In Progress' , 'Completed'] ,
                yticklabels=['Pending' , 'In Progress' , 'Completed'])
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_training_pipeline()
    print("\n✅ TRAINING AND ARTIFACT SAVING COMPLETED!")