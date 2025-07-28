"""
Main usage example demonstrating both classification and regression tasks
"""
import os  # Import the os module to access environment variables
import torch
import numpy as np

from src.data.models.task_dataset import TaskDatasetFactory, TaskDataProcessor
from src.data.preprocessing.pytorch_dataset import TaskModelFactory
import matplotlib.pyplot as plt


def main():
    # Database configuration using environment variables for Docker compatibility
    # Default values are provided for running outside of a containerized environment
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)), # Postgres default port
        'database': os.getenv('DB_NAME', 'licenta_db'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', '101102')
    }

    # Initialize data processor
    print("Initializing data processor...")
    data_processor = TaskDataProcessor(db_config)
    dataset_factory = TaskDatasetFactory(data_processor)

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # =================================
    # TASK 1: STATUS CLASSIFICATION
    # =================================
    print("\n" + "=" * 50)
    print("TASK 1: TASK STATUS CLASSIFICATION")
    print("=" * 50)

    # Create classification datasets
    print("Creating classification datasets...")
    classification_data = dataset_factory.create_classification_datasets(
        test_size=0.2,
        val_size=0.1,
        batch_size=32
    )

    if classification_data is None:
        print("Failed to create classification datasets. Is the database populated?")
        return

    # Create classification model
    print("Creating classification model...")
    classifier = TaskModelFactory.create_classifier(
        input_size=classification_data['num_features'],
        num_classes=classification_data['num_classes'],
        hidden_sizes=[128, 64, 32],
        dropout_rate=0.3
    )

    # Create trainer
    classifier_trainer = TaskModelFactory.create_trainer(classifier, device)

    # Train classification model
    print("Training classification model...")
    classifier_trainer.train_classification(
        train_loader=classification_data['train_loader'],
        val_loader=classification_data['val_loader'],
        epochs=100,
        lr=0.001,
        weight_decay=1e-5
    )

    # Evaluate classification model
    print("Evaluating classification model...")
    classification_results = classifier_trainer.evaluate_classification(
        classification_data['test_loader']
    )

    # Plot training history
    print("Plotting classification training history...")
    classifier_trainer.plot_training_history()

    # =================================
    # TASK 2: COMPLETION TIME REGRESSION
    # =================================
    print("\n" + "=" * 50)
    print("TASK 2: TASK COMPLETION TIME REGRESSION")
    print("=" * 50)

    # Create regression datasets
    print("Creating regression datasets...")
    regression_data = dataset_factory.create_regression_datasets(
        test_size=0.2,
        val_size=0.1,
        batch_size=32
    )

    if regression_data is None:
        print("Failed to create regression datasets. Is the database populated?")
        return

    # Create regression model
    print("Creating regression model...")
    regressor = TaskModelFactory.create_regressor(
        input_size=regression_data['num_features'],
        hidden_sizes=[128, 64, 32],
        dropout_rate=0.3
    )

    # Create trainer
    regressor_trainer = TaskModelFactory.create_trainer(regressor, device)

    # Train regression model
    print("Training regression model...")
    regressor_trainer.train_regression(
        train_loader=regression_data['train_loader'],
        val_loader=regression_data['val_loader'],
        epochs=100,
        lr=0.001,
        weight_decay=1e-5
    )

    # Evaluate regression model
    print("Evaluating regression model...")
    regression_results = regressor_trainer.evaluate_regression(
        regression_data['test_loader']
    )

    # Plot training history
    print("Plotting regression training history...")
    regressor_trainer.plot_training_history()

    # =================================
    # EXAMPLE PREDICTIONS
    # =================================
    print("\n" + "=" * 50)
    print("EXAMPLE PREDICTIONS")
    print("=" * 50)

    # Get a sample from test data for demonstration
    test_features, test_targets = next(iter(classification_data['test_loader']))
    sample_features = test_features[:5]  # Take first 5 samples

    # Classification predictions
    print("Classification predictions:")
    class_predictions, class_probabilities = classifier_trainer.predict(sample_features.numpy())

    # Map predictions back to status labels
    status_labels = {0: 'Pending', 1: 'In Progress', 2: 'Completed'}

    for i, (pred, probs) in enumerate(zip(class_predictions, class_probabilities)):
        print(f"Sample {i + 1}: Predicted Status = {status_labels[pred]}")
        print(f"  Probabilities: Pending={probs[0]:.3f}, In Progress={probs[1]:.3f}, Completed={probs[2]:.3f}")

    # Regression predictions
    print("\nRegression predictions:")
    time_predictions = regressor_trainer.predict(sample_features.numpy())

    for i, pred in enumerate(time_predictions):
        print(f"Sample {i + 1}: Predicted Completion Time = {pred:.2f} days")

    # =================================
    # SAVE MODELS AND PREPROCESSORS
    # =================================
    print("\n" + "=" * 50)
    print("SAVING MODELS AND PREPROCESSORS")
    print("=" * 50)

    # Save preprocessors
    data_processor.save_preprocessors('task_preprocessors.pkl')

    # Save model architectures and weights
    torch.save({
        'model_state_dict': classifier.state_dict(),
        'model_config': {
            'input_size': classification_data['num_features'],
            'num_classes': classification_data['num_classes'],
            'hidden_sizes': [128, 64, 32],
            'dropout_rate': 0.3
        }
    }, 'classification_model.pth')

    torch.save({
        'model_state_dict': regressor.state_dict(),
        'model_config': {
            'input_size': regression_data['num_features'],
            'hidden_sizes': [128, 64, 32],
            'dropout_rate': 0.3
        }
    }, 'regression_model.pth')

    print("Models and preprocessors saved successfully!")

    # =================================
    # RESULTS SUMMARY
    # =================================
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)

    print("Classification Results:")
    print(f"  Accuracy: {classification_results['accuracy']:.4f}")
    print(f"  Precision: {classification_results['precision']:.4f}")
    print(f"  Recall: {classification_results['recall']:.4f}")
    print(f"  F1-Score: {classification_results['f1']:.4f}")

    print("\nRegression Results:")
    print(f"  RMSE: {regression_results['rmse']:.4f} days")
    print(f"  R²: {regression_results['r2']:.4f}")
    print(f"  MAE: {regression_results['mae']:.4f} days")

    return {
        'classification_results': classification_results,
        'regression_results': regression_results,
        'classifier': classifier,
        'regressor': regressor,
        'data_processor': data_processor
    }


def load_and_predict_example():
    """
    Example of how to load saved models and make predictions on new data
    """
    print("\n" + "=" * 50)
    print("LOADING SAVED MODELS FOR PREDICTION")
    print("=" * 50)

    # Load preprocessors
    # An empty config is passed because we are only loading saved preprocessor objects,
    # which does not require a live database connection.
    data_processor = TaskDataProcessor({})
    data_processor.load_preprocessors('task_preprocessors.pkl')

    # Load classification model
    classifier_checkpoint = torch.load('classification_model.pth', map_location='cpu')
    classifier = TaskModelFactory.create_classifier(**classifier_checkpoint['model_config'])
    classifier.load_state_dict(classifier_checkpoint['model_state_dict'])

    # Load regression model
    regressor_checkpoint = torch.load('regression_model.pth', map_location='cpu')
    regressor = TaskModelFactory.create_regressor(**regressor_checkpoint['model_config'])
    regressor.load_state_dict(regressor_checkpoint['model_state_dict'])

    # Create trainers for prediction
    classifier_trainer = TaskModelFactory.create_trainer(classifier)
    regressor_trainer = TaskModelFactory.create_trainer(regressor)

    # Example: Create synthetic data for prediction
    # In practice, you would extract this from your database for new tasks
    synthetic_features = np.array([
        [5.0, 1, 9, 150, 12, 8, 0, 0, 1, 1, 1, 2, 1, 0],  # Example task 1
        [15.0, 5, 14, 50, 5, 15, 1, 1, 0, 0, 0, 1, 0, 1],  # Example task 2
    ])

    # Make predictions
    class_predictions, class_probabilities = classifier_trainer.predict(synthetic_features)
    time_predictions = regressor_trainer.predict(synthetic_features)

    status_labels = {0: 'Pending', 1: 'In Progress', 2: 'Completed'}

    print("Predictions for new tasks:")
    for i, (class_pred, class_probs, time_pred) in enumerate(
            zip(class_predictions, class_probabilities, time_predictions)):
        print(f"\nTask {i + 1}:")
        print(f"  Predicted Status: {status_labels[class_pred]}")
        print(f"  Status Probabilities: {dict(zip(status_labels.values(), class_probs))}")
        print(f"  Predicted Completion Time: {time_pred:.2f} days")


if __name__ == "__main__":
    # Run main training and evaluation
    results = main()

    # Only run the rest if main() was successful
    if results:
        # Demonstrate loading and prediction
        load_and_predict_example()

        print("\n" + "=" * 50)
        print("TRAINING AND EVALUATION COMPLETED!")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("MAIN PROCESS FAILED. SKIPPING PREDICTION EXAMPLE.")
        print("=" * 50)