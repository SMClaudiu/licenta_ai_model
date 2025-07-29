"""
Enhanced ML Pipeline Usage Example
Demonstrates how to use the improved models for better performance
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from src.data.models.task_dataset import TaskDataProcessor, TaskDatasetFactory
from src.data.preprocessing.pytorch_dataset import TaskModelFactory



def run_pipeline():

    #Database configuration

    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'licenta_db'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', '101102')
    }

    #Initialize data processor
    print("=" * 60)
    print("TASK MANAGEMENT ML PIPELINE")
    print("=" * 60)

    data_processor = TaskDataProcessor(db_config)
    dataset_factory = TaskDatasetFactory(data_processor)

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    print("\n" + "=" * 50)
    print("TASK STATUS CLASSIFICATION")
    print("=" * 50)

    print("Creating classification datasets...")
    classification_data = dataset_factory.create_classification_datasets(
        test_size=0.2,
        val_size=0.15,  # Slightly larger validation set
        batch_size=64,  # Larger batch size for better gradient estimates
        use_stratification=True
    )

    if classification_data is None:
        print("Failed to create classification datasets!")
        return None

    print("Creating classification model...")
    enhanced_classifier = TaskModelFactory.create_classifier(
        input_size=classification_data['num_features'],
        num_classes=classification_data['num_classes'],
        hidden_sizes=[256, 128, 64],  # Deeper network
        dropout_rate=0.4
    )

    #Create trainer with class weights
    class_distribution = classification_data['class_distribution']
    total_samples = sum(class_distribution.values())
    class_weights = [total_samples / (len(class_distribution) * count)
                    for count in class_distribution.values()]

    print(f"Using class weights: {class_weights}")

    classifier_trainer = TaskModelFactory.create_trainer(
        enhanced_classifier,
        device,
        class_weights=class_weights
    )

    print("Training classification model...")
    classifier_trainer.train_classification(
        train_loader=classification_data['train_loader'],
        val_loader=classification_data['val_loader'],
        epochs=150,  # More epochs with early stopping
        lr=0.001,
        weight_decay=1e-4
    )

    print("Evaluating classification model...")
    classification_results = classifier_trainer.evaluate_classification(
        classification_data['test_loader']
    )

    print("Plotting training history...")
    classifier_trainer.plot_enhanced_training_history()

    print("\n" + "=" * 50)
    print("TASK COMPLETION TIME REGRESSION")
    print("=" * 50)

    print("Creating enhanced regression datasets...")
    regression_data = dataset_factory.create_regression_datasets(
        test_size=0.2,
        val_size=0.15,
        batch_size=64
    )

    if regression_data is None:
        print("Failed to create regression datasets!")
        return None

    print("Creating enhanced regression model...")
    enhanced_regressor = TaskModelFactory.create_regressor(
        input_size=regression_data['num_features'],
        hidden_sizes=[256, 128, 64],
        dropout_rate=0.3
    )

    #Create trainer
    regressor_trainer = TaskModelFactory.create_trainer(enhanced_regressor, device)

    print("Training enhanced regression model...")
    regressor_trainer.train_regression(
        train_loader=regression_data['train_loader'],
        val_loader=regression_data['val_loader'],
        epochs=150,
        lr=0.001,
        weight_decay=1e-4
    )

    print("Evaluating enhanced regression model...")
    regression_results = regressor_trainer.evaluate_regression(
        regression_data['test_loader']
    )

    #History
    print("Plotting enhanced regression training history...")
    regressor_trainer.plot_enhanced_training_history()

    #Analysis + prediction
    print("\n" + "=" * 50)
    print("ANALYSIS AND PREDICTIONS")
    print("=" * 50)

    #Test samples
    test_features, test_targets = next(iter(classification_data['test_loader']))
    sample_features = test_features[:10].numpy()

    print("Classification Predictions:")
    class_predictions, class_probabilities, confidence_scores = classifier_trainer.predict(
        sample_features, return_confidence=True
    )

    status_labels = {0: 'Pending', 1: 'In Progress', 2: 'Completed'}

    for i, (pred, probs, conf) in enumerate(zip(class_predictions, class_probabilities, confidence_scores)):
        print(f"Sample {i + 1}:")
        print(f"  Predicted Status: {status_labels[pred]} (Confidence: {conf:.3f})")
        print(f"  Probabilities: {dict(zip(status_labels.values(), probs))}")
        print()

    print("Regression Predictions:")
    time_predictions = regressor_trainer.predict(sample_features)

    for i, pred in enumerate(time_predictions):
        print(f"Sample {i + 1}: Predicted Completion Time = {pred:.2f} days")

    print("\n" + "=" * 50)
    print("MODEL INTERPRETATION")
    print("=" * 50)

    # Feature importance analysis (simplified)f
    print("Most important features for classification:")
    feature_names = classification_data['feature_names']
    print(f"Total selected features: {len(feature_names)}")
    for i, feature in enumerate(feature_names[:10]):  # Top 10
        print(f"  {i+1}. {feature}")

    print("\n" + "=" * 50)
    print("SAVING MODELS")
    print("=" * 50)

    data_processor.save_preprocessors('enhanced_task_preprocessors.pkl')

    torch.save({
        'model_state_dict': enhanced_classifier.state_dict(),
        'model_config': {
            'input_size': classification_data['num_features'],
            'num_classes': classification_data['num_classes'],
            'hidden_sizes': [256, 128, 64],
            'dropout_rate': 0.4
        },
        'feature_names': feature_names,
        'class_weights': class_weights
    }, 'enhanced_classification_model.pth')

    torch.save({
        'model_state_dict': enhanced_regressor.state_dict(),
        'model_config': {
            'input_size': regression_data['num_features'],
            'hidden_sizes': [256, 128, 64],
            'dropout_rate': 0.3
        },
        'feature_names': regression_data['feature_names']
    }, 'enhanced_regression_model.pth')

    print("Enhanced models and preprocessors saved successfully!")

    #Performance comparison
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)

    print("Enhanced Classification Results:")
    print(f"  Overall Accuracy: {classification_results['accuracy']:.4f}")
    print(f"  Macro Precision: {classification_results['precision']:.4f}")
    print(f"  Macro Recall: {classification_results['recall']:.4f}")
    print(f"  Macro F1-Score: {classification_results['f1']:.4f}")

    print("\nPer-class Performance:")
    class_names = ['Pending', 'In Progress', 'Completed']
    for i, class_name in enumerate(class_names):
        if i < len(classification_results['per_class_precision']):
            print(f"  {class_name}:")
            print(f"    Precision: {classification_results['per_class_precision'][i]:.4f}")
            print(f"    Recall: {classification_results['per_class_recall'][i]:.4f}")
            print(f"    F1-Score: {classification_results['per_class_f1'][i]:.4f}")

    print(f"\nEnhanced Regression Results:")
    print(f"  RMSE: {regression_results['rmse']:.4f} days")
    print(f"  R²: {regression_results['r2']:.4f}")
    print(f"  MAE: {regression_results['mae']:.4f} days")
    print(f"  MSE: {regression_results['mse']:.4f}")

    create_confusion_matrix_plot(
        classification_results['targets'],
        classification_results['predictions']
    )

    return {
        'classification_results': classification_results,
        'regression_results': regression_results,
        'enhanced_classifier': enhanced_classifier,
        'enhanced_regressor': enhanced_regressor,
        'data_processor': data_processor
    }


def create_confusion_matrix_plot(y_true, y_pred):
    """Create and display confusion matrix"""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=['Pending', 'In Progress', 'Completed'],
                yticklabels=['Pending', 'In Progress', 'Completed'])

    plt.title('Normalized Confusion Matrix - Enhanced Classification Model')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


def load_and_predict() -> None:
    #ToRecheck
    #Example of loading enhanced models for prediction
    print("\n" + "=" * 50)
    print("LOADING ENHANCED MODELS FOR PREDICTION")
    print("=" * 50)

    #Load preprocessors
    data_processor = TaskDataProcessor({})
    data_processor.load_preprocessors('enhanced_task_preprocessors.pkl')

    #Load classification model
    classifier_checkpoint = torch.load('enhanced_classification_model.pth', map_location='cpu')
    enhanced_classifier = TaskModelFactory.create_classifier(**classifier_checkpoint['model_config'])
    enhanced_classifier.load_state_dict(classifier_checkpoint['model_state_dict'])

    #Load regression model
    regressor_checkpoint = torch.load('enhanced_regression_model.pth', map_location='cpu')
    enhanced_regressor = TaskModelFactory.create_regressor(**regressor_checkpoint['model_config'])
    enhanced_regressor.load_state_dict(regressor_checkpoint['model_state_dict'])

    #Trainers for prediction
    classifier_trainer = TaskModelFactory.create_trainer(enhanced_classifier)
    regressor_trainer = TaskModelFactory.create_trainer(enhanced_regressor)

    #ToRecheck
    #Task scenarios
    synthetic_features = np.array([
        # High priority, short deadline task
        [0.2, 1, 9, 3, 1, 80, 25, 15, 3, 5, 7.5, 25, 10, 12.5, 0, 0, 1, 1, 0,
         0.8, 0.6, 0.1, 0.9, 0.5, 0.8, 1, 0, 0, 1, 1, 1, 0.6, 0.4, 0.8, 0.7,
         2.1, 0.9, 15, 18, 25, 120, 85, 1.2, 0.8, 3, 2, 8.5],

        # Regular task, medium complexity
        [0.5, 3, 14, 6, 2, 150, 35, 25, 8, 12, 15.0, 45, 20, 18.0, 0, 1, 0, 0, 0,
         0.2, -0.4, 0.7, -0.2, 0.0, 1.0, 0, 1, 0, 0, 0, 0, 0.75, 0.25, 0.65, 0.6,
         1.8, 1.1, 35, 28, 15, 85, 45, 0.9, 1.1, 10, 5, 4.2],

        # Long-term project task
        [0.1, 0, 10, 12, 4, 300, 45, 35, 15, 20, 45.0, 60, 35, 38.0, 0, 0, 1, 0, 1,
         -0.8, 0.6, 1.0, 0.0, 0.0, 0.0, 0, 0, 1, 0, 0, 1, 0.85, 0.15, 0.9, 0.8,
         1.2, 1.3, 8, 12, 8, 45, 32, 0.7, 1.2, 45, 30, 1.8]
    ])

    #Ensure the data has the right number of features
    expected_features = len(classifier_checkpoint.get('feature_names', []))
    if synthetic_features.shape[1] != expected_features:
        print(f"Adjusting synthetic data from {synthetic_features.shape[1]} to {expected_features} features")
        if synthetic_features.shape[1] > expected_features:
            synthetic_features = synthetic_features[:, :expected_features]
        else:
            padding = np.zeros((synthetic_features.shape[0], expected_features - synthetic_features.shape[1]))
            synthetic_features = np.concatenate([synthetic_features, padding], axis=1)

    print("Enhanced predictions for new tasks:")

    #Classification with confidence
    class_predictions, class_probabilities, confidence_scores = classifier_trainer.predict(
        synthetic_features, return_confidence=True
    )

    #Regression predictions
    time_predictions = regressor_trainer.predict(synthetic_features)

    status_labels = {0: 'Pending', 1: 'In Progress', 2: 'Completed'}
    task_descriptions = [
        "High Priority Short Deadline Task",
        "Regular Medium Complexity Task",
        "Long-term Project Task"
    ]

    for i, (desc, class_pred, class_probs, conf, time_pred) in enumerate(
            zip(task_descriptions, class_predictions, class_probabilities, confidence_scores, time_predictions)):
        print(f"\n{desc}:")
        print(f"  Predicted Status: {status_labels[class_pred]} (Confidence: {conf:.3f})")
        print(f"  Status Probabilities:")
        for status, prob in zip(status_labels.values(), class_probs):
            print(f"    {status}: {prob:.3f}")
        print(f"  Predicted Completion Time: {time_pred:.2f} days")

        #Add interpretation
        if conf < 0.6:
            print(f"  ⚠️  Low confidence prediction - consider manual review")
        if time_pred > 30:
            print(f"  📅 Long-term task - consider breaking into smaller tasks")
        if class_pred == 1 and conf > 0.8:
            print(f"  ✅ High confidence 'In Progress' - likely actively worked on")



if __name__ == "__main__":
    print("🚀 Starting Enhanced ML Pipeline...")

    # print(torch.version.cuda)
    # print(torch.cuda.is_available())

    results = run_pipeline()

    if results:

        try:
            load_and_predict()
        except FileNotFoundError:
            print("Enhanced model files not found. Run the main pipeline first.")

        print("✅ ENHANCED TRAINING AND EVALUATION COMPLETED!")

    else:
        print("\n❌ ENHANCED PIPELINE FAILED")
        print("Check database connection and data quality")

        print("\n🔧 TROUBLESHOOTING TIPS:")
        print("1. Verify database is populated with sufficient data")
        print("2. Check for data quality issues (nulls, outliers)")
        print("3. Ensure all required features are present")
        print("4. Validate data types and encoding")
        print("5. Check for class imbalance in target variables")