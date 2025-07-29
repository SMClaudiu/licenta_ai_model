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

# Import the enhanced classes
from src.data.models.task_dataset import EnhancedTaskDataProcessor, EnhancedTaskDatasetFactory
from src.data.preprocessing.pytorch_dataset import EnhancedTaskModelFactory



def run_enhanced_pipeline():
    """Main function demonstrating the enhanced ML pipeline"""

    # Database configuration
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'licenta_db'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', '101102')
    }

    # Initialize enhanced data processor
    print("=" * 60)
    print("ENHANCED TASK MANAGEMENT ML PIPELINE")
    print("=" * 60)

    data_processor = EnhancedTaskDataProcessor(db_config)
    dataset_factory = EnhancedTaskDatasetFactory(data_processor)

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ========================================
    # ENHANCED CLASSIFICATION TASK
    # ========================================
    print("\n" + "=" * 50)
    print("ENHANCED TASK STATUS CLASSIFICATION")
    print("=" * 50)

    # Create enhanced classification datasets
    print("Creating enhanced classification datasets...")
    classification_data = dataset_factory.create_classification_datasets(
        test_size=0.2,
        val_size=0.15,  # Slightly larger validation set
        batch_size=64,  # Larger batch size for better gradient estimates
        use_stratification=True
    )

    if classification_data is None:
        print("Failed to create classification datasets!")
        return None

    # Create enhanced classification model
    print("Creating enhanced classification model...")
    enhanced_classifier = EnhancedTaskModelFactory.create_classifier(
        input_size=classification_data['num_features'],
        num_classes=classification_data['num_classes'],
        hidden_sizes=[256, 128, 64],  # Deeper network
        dropout_rate=0.4
    )

    # Create enhanced trainer with class weights
    class_distribution = classification_data['class_distribution']
    total_samples = sum(class_distribution.values())
    class_weights = [total_samples / (len(class_distribution) * count)
                    for count in class_distribution.values()]

    print(f"Using class weights: {class_weights}")

    classifier_trainer = EnhancedTaskModelFactory.create_trainer(
        enhanced_classifier,
        device,
        class_weights=class_weights
    )

    # Enhanced training
    print("Training enhanced classification model...")
    classifier_trainer.train_classification(
        train_loader=classification_data['train_loader'],
        val_loader=classification_data['val_loader'],
        epochs=150,  # More epochs with early stopping
        lr=0.001,
        weight_decay=1e-4
    )

    # Enhanced evaluation
    print("Evaluating enhanced classification model...")
    classification_results = classifier_trainer.evaluate_classification(
        classification_data['test_loader']
    )

    # Plot enhanced training history
    print("Plotting enhanced training history...")
    classifier_trainer.plot_enhanced_training_history()

    # ========================================
    # ENHANCED REGRESSION TASK
    # ========================================
    print("\n" + "=" * 50)
    print("ENHANCED TASK COMPLETION TIME REGRESSION")
    print("=" * 50)

    # Create enhanced regression datasets
    print("Creating enhanced regression datasets...")
    regression_data = dataset_factory.create_regression_datasets(
        test_size=0.2,
        val_size=0.15,
        batch_size=64
    )

    if regression_data is None:
        print("Failed to create regression datasets!")
        return None

    # Create enhanced regression model
    print("Creating enhanced regression model...")
    enhanced_regressor = EnhancedTaskModelFactory.create_regressor(
        input_size=regression_data['num_features'],
        hidden_sizes=[256, 128, 64],
        dropout_rate=0.3
    )

    # Create enhanced trainer
    regressor_trainer = EnhancedTaskModelFactory.create_trainer(enhanced_regressor, device)

    # Enhanced training
    print("Training enhanced regression model...")
    regressor_trainer.train_regression(
        train_loader=regression_data['train_loader'],
        val_loader=regression_data['val_loader'],
        epochs=150,
        lr=0.001,
        weight_decay=1e-4
    )

    # Enhanced evaluation
    print("Evaluating enhanced regression model...")
    regression_results = regressor_trainer.evaluate_regression(
        regression_data['test_loader']
    )

    # Plot enhanced training history
    print("Plotting enhanced regression training history...")
    regressor_trainer.plot_enhanced_training_history()

    # ========================================
    # ADVANCED ANALYSIS AND PREDICTIONS
    # ========================================
    print("\n" + "=" * 50)
    print("ADVANCED PREDICTIONS AND ANALYSIS")
    print("=" * 50)

    # Get test samples for detailed analysis
    test_features, test_targets = next(iter(classification_data['test_loader']))
    sample_features = test_features[:10].numpy()

    # Enhanced classification predictions with confidence
    print("Enhanced Classification Predictions:")
    class_predictions, class_probabilities, confidence_scores = classifier_trainer.predict(
        sample_features, return_confidence=True
    )

    status_labels = {0: 'Pending', 1: 'In Progress', 2: 'Completed'}

    for i, (pred, probs, conf) in enumerate(zip(class_predictions, class_probabilities, confidence_scores)):
        print(f"Sample {i + 1}:")
        print(f"  Predicted Status: {status_labels[pred]} (Confidence: {conf:.3f})")
        print(f"  Probabilities: {dict(zip(status_labels.values(), probs))}")
        print()

    # Enhanced regression predictions
    print("Enhanced Regression Predictions:")
    time_predictions = regressor_trainer.predict(sample_features)

    for i, pred in enumerate(time_predictions):
        print(f"Sample {i + 1}: Predicted Completion Time = {pred:.2f} days")

    # ========================================
    # MODEL INTERPRETATION
    # ========================================
    print("\n" + "=" * 50)
    print("MODEL INTERPRETATION")
    print("=" * 50)

    # Feature importance analysis (simplified)f
    print("Most important features for classification:")
    feature_names = classification_data['feature_names']
    print(f"Total selected features: {len(feature_names)}")
    for i, feature in enumerate(feature_names[:10]):  # Top 10
        print(f"  {i+1}. {feature}")

    # ========================================
    # SAVE ENHANCED MODELS
    # ========================================
    print("\n" + "=" * 50)
    print("SAVING ENHANCED MODELS")
    print("=" * 50)

    # Save enhanced preprocessors
    data_processor.save_preprocessors('enhanced_task_preprocessors.pkl')

    # Save enhanced models
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

    # ========================================
    # PERFORMANCE COMPARISON
    # ========================================
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

    # Create confusion matrix visualization
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


def load_and_predict_enhanced():
    """Example of loading enhanced models for prediction"""
    print("\n" + "=" * 50)
    print("LOADING ENHANCED MODELS FOR PREDICTION")
    print("=" * 50)

    # Load enhanced preprocessors
    data_processor = EnhancedTaskDataProcessor({})
    data_processor.load_preprocessors('enhanced_task_preprocessors.pkl')

    # Load enhanced classification model
    classifier_checkpoint = torch.load('enhanced_classification_model.pth', map_location='cpu')
    enhanced_classifier = EnhancedTaskModelFactory.create_classifier(**classifier_checkpoint['model_config'])
    enhanced_classifier.load_state_dict(classifier_checkpoint['model_state_dict'])

    # Load enhanced regression model
    regressor_checkpoint = torch.load('enhanced_regression_model.pth', map_location='cpu')
    enhanced_regressor = EnhancedTaskModelFactory.create_regressor(**regressor_checkpoint['model_config'])
    enhanced_regressor.load_state_dict(regressor_checkpoint['model_state_dict'])

    # Create enhanced trainers for prediction
    classifier_trainer = EnhancedTaskModelFactory.create_trainer(enhanced_classifier)
    regressor_trainer = EnhancedTaskModelFactory.create_trainer(enhanced_regressor)

    # Example: Create more realistic synthetic data for prediction
    # This represents various task scenarios
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

    # Ensure the synthetic data has the right number of features
    expected_features = len(classifier_checkpoint.get('feature_names', []))
    if synthetic_features.shape[1] != expected_features:
        print(f"Adjusting synthetic data from {synthetic_features.shape[1]} to {expected_features} features")
        if synthetic_features.shape[1] > expected_features:
            synthetic_features = synthetic_features[:, :expected_features]
        else:
            # Pad with zeros if needed
            padding = np.zeros((synthetic_features.shape[0], expected_features - synthetic_features.shape[1]))
            synthetic_features = np.concatenate([synthetic_features, padding], axis=1)

    # Make enhanced predictions
    print("Enhanced predictions for new tasks:")

    # Classification with confidence
    class_predictions, class_probabilities, confidence_scores = classifier_trainer.predict(
        synthetic_features, return_confidence=True
    )

    # Regression predictions
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

        # Add interpretation
        if conf < 0.6:
            print(f"  ⚠️  Low confidence prediction - consider manual review")
        if time_pred > 30:
            print(f"  📅 Long-term task - consider breaking into smaller tasks")
        if class_pred == 1 and conf > 0.8:
            print(f"  ✅ High confidence 'In Progress' - likely actively worked on")


def compare_model_performance():
    """Compare original vs enhanced model performance"""
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 60)

    # Expected improvements based on the enhancements
    print("Expected Improvements with Enhanced Models:")
    print("\n🔍 CLASSIFICATION IMPROVEMENTS:")
    print("  • Original Accuracy: ~35.2% → Enhanced Expected: 60-75%")
    print("  • Better class balance handling with weighted sampling")
    print("  • Confidence scores for prediction reliability")
    print("  • Improved architecture with batch normalization")
    print("  • Advanced feature engineering (40+ features)")

    print("\n📊 REGRESSION IMPROVEMENTS:")
    print("  • Original R²: 0.9812 → Enhanced Expected: 0.985+ (marginal)")
    print("  • Better outlier handling with Huber loss")
    print("  • More robust feature selection")
    print("  • Improved generalization with advanced regularization")

    print("\n🚀 GENERAL IMPROVEMENTS:")
    print("  • Feature selection reduces overfitting")
    print("  • Cyclical encoding for time-based features")
    print("  • Interaction features capture complex relationships")
    print("  • Robust scaling handles outliers better")
    print("  • Enhanced early stopping prevents overfitting")
    print("  • Learning rate scheduling for better convergence")


def generate_model_insights():
    """Generate insights about model behavior and recommendations"""
    print("\n" + "=" * 60)
    print("MODEL INSIGHTS AND RECOMMENDATIONS")
    print("=" * 60)

    print("📈 KEY INSIGHTS FROM YOUR CURRENT RESULTS:")
    print("\n1. CLASSIFICATION CHALLENGES:")
    print("   • Low accuracy (35%) suggests class imbalance or insufficient features")
    print("   • Flat probability distributions indicate model uncertainty")
    print("   • Training instability visible in the metrics plot")

    print("\n2. REGRESSION SUCCESS:")
    print("   • Excellent R² (98.12%) shows strong predictive power")
    print("   • Stable training convergence")
    print("   • Good generalization to test data")

    print("\n3. POTENTIAL DATA ISSUES:")
    print("   • Classification may have overlapping classes")
    print("   • Need more discriminative features")
    print("   • Possible data quality issues")

    print("\n🛠️  RECOMMENDED IMPROVEMENTS:")
    print("\n1. DATA ENHANCEMENTS:")
    print("   ✅ Add temporal patterns (seasonality, trends)")
    print("   ✅ Include user behavior metrics")
    print("   ✅ Add task dependency information")
    print("   ✅ Include priority and urgency indicators")

    print("\n2. MODEL ARCHITECTURE:")
    print("   ✅ Use ensemble methods (Random Forest + Neural Network)")
    print("   ✅ Implement attention mechanisms for feature importance")
    print("   ✅ Add residual connections for deeper networks")

    print("\n3. TRAINING STRATEGIES:")
    print("   ✅ Use focal loss for class imbalance")
    print("   ✅ Implement progressive resizing")
    print("   ✅ Add label smoothing for regularization")

    print("\n4. EVALUATION IMPROVEMENTS:")
    print("   ✅ Use cross-validation for robust estimates")
    print("   ✅ Add business metrics (cost-sensitive evaluation)")
    print("   ✅ Implement A/B testing framework")


if __name__ == "__main__":
    print("🚀 Starting Enhanced ML Pipeline...")

    # print(torch.version.cuda)
    # print(torch.cuda.is_available())

    # Run the enhanced pipeline
    results = run_enhanced_pipeline()

    if results:
        # Compare performance
        compare_model_performance()

        # Generate insights
        generate_model_insights()

        # Demonstrate loading and prediction
        try:
            load_and_predict_enhanced()
        except FileNotFoundError:
            print("Enhanced model files not found. Run the main pipeline first.")

        print("\n" + "=" * 60)
        print("✅ ENHANCED TRAINING AND EVALUATION COMPLETED!")
        print("=" * 60)

        print("\n📊 NEXT STEPS:")
        print("1. Monitor model performance in production")
        print("2. Collect user feedback on predictions")
        print("3. Retrain models with new data regularly")
        print("4. Implement model versioning and A/B testing")
        print("5. Add explainability features for business users")

    else:
        print("\n❌ ENHANCED PIPELINE FAILED")
        print("Check database connection and data quality")

        print("\n🔧 TROUBLESHOOTING TIPS:")
        print("1. Verify database is populated with sufficient data")
        print("2. Check for data quality issues (nulls, outliers)")
        print("3. Ensure all required features are present")
        print("4. Validate data types and encoding")
        print("5. Check for class imbalance in target variables")