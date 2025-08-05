# pipeline.py (Varianta Finală, Îmbunătățită)

import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Presupunem că aceste importuri sunt corecte relativ la structura proiectului tău
from src.data.models.task_dataset import TaskDataProcessor, TaskDatasetFactory
from src.data.preprocessing.pytorch_dataset import TaskModelFactory

def run_pipeline():
    """
    Funcția principală de orchestrare. Nu a necesitat modificări,
    deoarece este deja bine structurată.
    """
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'), 'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'licenta_db'), 'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', '101102')
    }
    print("=" * 60); print("TASK MANAGEMENT ML PIPELINE"); print("=" * 60)
    data_processor = TaskDataProcessor(db_config)
    dataset_factory = TaskDatasetFactory(data_processor)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- CLASIFICARE ---
    print("\n" + "=" * 50); print("TASK STATUS CLASSIFICATION"); print("=" * 50)
    print("Creating classification datasets...")
    classification_data = dataset_factory.create_classification_datasets(test_size=0.2, val_size=0.15, batch_size=64, use_stratification=True)
    if classification_data is None: return None
    print("Creating classification model...")
    enhanced_classifier = TaskModelFactory.create_classifier(input_size=classification_data['num_features'], num_classes=classification_data['num_classes'])
    class_distribution = classification_data['class_distribution']
    total_samples = sum(class_distribution.values())
    class_weights = [total_samples / (len(class_distribution) * count) for count in class_distribution.values()]
    print(f"Using class weights: {class_weights}")
    classifier_trainer = TaskModelFactory.create_trainer(enhanced_classifier, device, class_weights=class_weights)
    print("Training classification model...")
    classifier_trainer.train_classification(train_loader=classification_data['train_loader'], val_loader=classification_data['val_loader'], epochs=150)
    print("Evaluating classification model...")
    classification_results = classifier_trainer.evaluate_classification(classification_data['test_loader'])
    print("Plotting training history...")
    classifier_trainer.plot_enhanced_training_history()

    # --- REGRESIE ---
    print("\n" + "=" * 50); print("TASK COMPLETION TIME REGRESSION"); print("=" * 50)
    print("Creating enhanced regression datasets...")
    regression_data = dataset_factory.create_regression_datasets(test_size=0.2, val_size=0.15, batch_size=64)
    if regression_data is None: return None
    print("Creating enhanced regression model...")
    enhanced_regressor = TaskModelFactory.create_regressor(input_size=regression_data['num_features'])
    regressor_trainer = TaskModelFactory.create_trainer(enhanced_regressor, device)
    print("Training enhanced regression model...")
    regressor_trainer.train_regression(train_loader=regression_data['train_loader'], val_loader=regression_data['val_loader'], epochs=150)
    print("Evaluating enhanced regression model...")
    regression_results = regressor_trainer.evaluate_regression(regression_data['test_loader'])
    print("Plotting enhanced regression training history...")
    regressor_trainer.plot_enhanced_training_history()

    # --- SALVARE ȘI REZUMAT ---
    print("\n" + "=" * 50); print("SAVING MODELS"); print("=" * 50)
    data_processor.save_preprocessors('enhanced_task_preprocessors.pkl')
    torch.save({'model_state_dict': enhanced_classifier.state_dict(), 'model_config': {'input_size': classification_data['num_features'], 'num_classes': classification_data['num_classes']}, 'feature_names': classification_data['feature_names']}, 'enhanced_classification_model.pth')
    torch.save({'model_state_dict': enhanced_regressor.state_dict(), 'model_config': {'input_size': regression_data['num_features']}, 'feature_names': regression_data['feature_names']}, 'enhanced_regression_model.pth')
    print("Enhanced models and preprocessors saved successfully!")

    print("\n" + "=" * 50); print("PERFORMANCE SUMMARY"); print("=" * 50)
    print("Enhanced Classification Results:"); print(f"  Overall Accuracy: {classification_results['accuracy']:.4f}"); print(f"  Macro F1-Score: {classification_results['f1']:.4f}")
    create_confusion_matrix_plot(classification_results['targets'], classification_results['predictions'])
    print(f"\nEnhanced Regression Results:"); print(f"  RMSE: {regression_results['rmse']:.4f} days"); print(f"  R²: {regression_results['r2']:.4f}")

    return {'data_processor': data_processor, 'enhanced_classifier': enhanced_classifier, 'enhanced_regressor': enhanced_regressor}


def create_confusion_matrix_plot(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', xticklabels=['Pending', 'In Progress', 'Completed'], yticklabels=['Pending', 'In Progress', 'Completed'])
    plt.title('Normalized Confusion Matrix - Enhanced Classification Model'); plt.xlabel('Predicted Label'); plt.ylabel('True Label')
    plt.tight_layout(); plt.show()


def load_and_predict():
    """
    MODIFICAT COMPLET: Simulează corect pipeline-ul de predicție pentru task-uri noi,
    folosind pre-procesoarele salvate pentru a evita nealinierea caracteristicilor.
    """
    print("\n" + "=" * 50); print("LOADING ENHANCED MODELS FOR PREDICTION"); print("=" * 50)

    data_processor = TaskDataProcessor({})
    data_processor.load_preprocessors('enhanced_task_preprocessors.pkl')

    classifier_checkpoint = torch.load('enhanced_classification_model.pth', map_location='cpu')
    enhanced_classifier = TaskModelFactory.create_classifier(**classifier_checkpoint['model_config'])
    enhanced_classifier.load_state_dict(classifier_checkpoint['model_state_dict'])
    classifier_trainer = TaskModelFactory.create_trainer(enhanced_classifier)

    regressor_checkpoint = torch.load('enhanced_regression_model.pth', map_location='cpu')
    enhanced_regressor = TaskModelFactory.create_regressor(**regressor_checkpoint['model_config'])
    enhanced_regressor.load_state_dict(regressor_checkpoint['model_state_dict'])
    regressor_trainer = TaskModelFactory.create_trainer(enhanced_regressor)

    task_scenarios = [
        {"task_name": "URGENT: Fix login button", "description": "Critical bug reported.", "creation_date": datetime.now() - timedelta(hours=1), "due_date": datetime.now() + timedelta(days=1), "client_total_tasks": 50, "client_completed_tasks": 40, "client_pending_tasks": 2, "client_avg_task_duration": 5.5, "board_total_tasks": 100, "board_completed_tasks": 80, "board_avg_task_duration": 4.0, "board_name": "Production Hotfixes", "client_name": "Acme Corp", "email": "contact@acme.com"},
        {"task_name": "Draft Q4 marketing report", "description": "Prepare the quarterly report.", "creation_date": datetime.now() - timedelta(days=2), "due_date": datetime.now() + timedelta(days=12), "client_total_tasks": 20, "client_completed_tasks": 15, "client_pending_tasks": 3, "client_avg_task_duration": 8.0, "board_total_tasks": 40, "board_completed_tasks": 30, "board_avg_task_duration": 10.0, "board_name": "Content Calendar - Blog", "client_name": "Innovate LLC", "email": "hello@innovate.com"},
        {"task_name": "Research new markets", "description": "Long-term strategic task.", "creation_date": datetime.now() - timedelta(days=10), "due_date": datetime.now() + timedelta(days=50), "client_total_tasks": 0, "client_completed_tasks": 0, "client_pending_tasks": 0, "client_avg_task_duration": 0, "board_total_tasks": 5, "board_completed_tasks": 1, "board_avg_task_duration": 25.0, "board_name": "New Market Expansion", "client_name": "Future Ventures", "email": "info@future.com"}
    ]

    df_predict = pd.DataFrame(task_scenarios)

    # Adăugăm manual coloanele simple necesare pentru `engineer_advanced_features`
    df_predict['days_to_complete'] = (df_predict['due_date'] - df_predict['creation_date']).dt.days
    df_predict['creation_day_of_week'] = df_predict['creation_date'].dt.dayofweek
    df_predict['creation_hour'] = df_predict['creation_date'].dt.hour
    df_predict['creation_month'] = df_predict['creation_date'].dt.month
    df_predict['creation_quarter'] = df_predict['creation_date'].dt.quarter
    df_predict['description_length'] = df_predict['description'].str.len()
    df_predict['task_name_length'] = df_predict['task_name'].str.len()

    df_processed = data_processor.engineer_advanced_features(df_predict)

    final_features_list = data_processor.feature_names
    df_final = df_processed[final_features_list].fillna(0)

    scaled_features = data_processor.scaler.transform(df_final)

    print("Enhanced predictions for new tasks:")
    class_predictions, class_probabilities, confidence_scores = classifier_trainer.predict(scaled_features, return_confidence=True)
    time_predictions = regressor_trainer.predict(scaled_features)

    status_labels = {0: 'Pending', 1: 'In Progress', 2: 'Completed'}
    task_descriptions = ["High Priority Short Deadline Task", "Regular Medium Complexity Task", "Long-term Project Task (Cold Start)"]

    for i, desc in enumerate(task_descriptions):
        print(f"\n{desc}:")
        print(f"  Predicted Status: {status_labels[class_predictions[i]]} (Confidence: {confidence_scores[i]:.3f})")
        print(f"  Predicted Completion Time: {time_predictions[i]:.2f} days")


if __name__ == "__main__":
    print("🚀 Starting Enhanced ML Pipeline...")
    results = run_pipeline()
    if results:
        try:
            load_and_predict()
        except FileNotFoundError:
            print("Enhanced model files not found. Run the main pipeline first.")
        print("✅ ENHANCED TRAINING AND EVALUATION COMPLETED!")
    else:
        print("\n❌ ENHANCED PIPELINE FAILED")