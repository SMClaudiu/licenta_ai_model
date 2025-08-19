import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime , timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import shap

from src.data.models.task_dataset import TaskDataProcessor , TaskDatasetFactory
from src.data.preprocessing.pytorch_dataset import TaskModelFactory


def run_pipeline():
    db_config = {
        'host':os.getenv('DB_HOST' , 'localhost') , 'port':int(os.getenv('DB_PORT' , 5432)) ,
        'database':os.getenv('DB_NAME' , 'licenta_db') , 'user':os.getenv('DB_USER' , 'postgres') ,
        'password':os.getenv('DB_PASSWORD' , '101102')
    }
    print("=" * 60);
    print("TASK MANAGEMENT ML PIPELINE");
    print("=" * 60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n" + "=" * 50);
    print("TASK STATUS CLASSIFICATION");
    print("=" * 50)
    classification_processor = TaskDataProcessor(db_config)
    classification_dataset_factory = TaskDatasetFactory(classification_processor)
    print("Creating classification datasets...")
    classification_data = classification_dataset_factory.create_classification_datasets(test_size=0.2 , val_size=0.15 ,
                                                                                        batch_size=64 ,
                                                                                        use_stratification=True)
    if classification_data is None: return None
    print("Creating classification model...")
    classifier = TaskModelFactory.create_classifier(input_size=classification_data['num_features'] ,
                                                    num_classes=classification_data['num_classes'])
    class_distribution = classification_data['class_distribution']
    total_samples = sum(class_distribution.values())
    class_weights = [total_samples / (len(class_distribution) * count) for count in class_distribution.values()]
    print(f"Using class weights: {class_weights}")
    classifier_trainer = TaskModelFactory.create_trainer(classifier , device , class_weights=class_weights)
    print("Training classification model...")
    classifier_trainer.train_classification(train_loader=classification_data['train_loader'] ,
                                            val_loader=classification_data['val_loader'] , epochs=150)
    print("Evaluating classification model...")
    classification_results = classifier_trainer.evaluate_classification(classification_data['test_loader'])
    classifier_trainer.plot_enhanced_training_history()

    print("\n" + "=" * 50);
    print("TASK COMPLETION TIME REGRESSION");
    print("=" * 50)
    regression_processor = TaskDataProcessor(db_config)
    regression_dataset_factory = TaskDatasetFactory(regression_processor)
    print("Creating regression datasets...")
    regression_data = regression_dataset_factory.create_regression_datasets(test_size=0.2 , val_size=0.15 ,
                                                                            batch_size=64)
    if regression_data is None: return None
    print("Creating regression model...")
    regressor = TaskModelFactory.create_regressor(input_size=regression_data['num_features'])
    regressor_trainer = TaskModelFactory.create_trainer(regressor , device)
    print("Training regression model...")
    regressor_trainer.train_regression(train_loader=regression_data['train_loader'] ,
                                       val_loader=regression_data['val_loader'] , epochs=150)
    print("Evaluating regression model...")
    regression_results = regressor_trainer.evaluate_regression(regression_data['test_loader'])
    regressor_trainer.plot_enhanced_training_history()

    print("\n" + "=" * 50);
    print("SAVING MODELS AND PREPROCESSORS");
    print("=" * 50)
    classification_processor.save_preprocessors('classification_preprocessors.pkl')
    regression_processor.save_preprocessors('regression_preprocessors.pkl')
    torch.save({
                   'model_state_dict':classifier.state_dict() , 'model_config':{
            'input_size':classification_data['num_features'] , 'num_classes':classification_data['num_classes'] } } ,
               'classification_model.pth')
    torch.save({
                   'model_state_dict':regressor.state_dict() ,
                   'model_config':{ 'input_size':regression_data['num_features'] } } , 'regression_model.pth')
    print("Models and dedicated preprocessors saved successfully!")

    print("\n" + "=" * 50);
    print("PERFORMANCE SUMMARY");
    print("=" * 50)
    print("Classification Results:");
    print(f"  Overall Accuracy: {classification_results['accuracy']:.4f}");
    print(f"  Macro F1-Score: {classification_results['f1']:.4f}")
    create_confusion_matrix_plot(classification_results['targets'] , classification_results['predictions'])
    print(f"\nRegression Results:");
    print(f"  RMSE: {regression_results['rmse']:.4f} days");
    print(f"  R²: {regression_results['r2']:.4f}")
    return { 'classifier':classifier , 'regressor':regressor }


def create_confusion_matrix_plot(y_true , y_pred):
    plt.figure(figsize=(8 , 6))
    cm = confusion_matrix(y_true , y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[: , np.newaxis]
    sns.heatmap(cm_normalized , annot=True , fmt='.3f' , cmap='Blues' ,
                xticklabels=['Pending' , 'In Progress' , 'Completed'] ,
                yticklabels=['Pending' , 'In Progress' , 'Completed'])
    plt.title('Normalized Confusion Matrix');
    plt.xlabel('Predicted Label');
    plt.ylabel('True Label')
    plt.tight_layout();
    plt.show()


def load_and_predict():
    print("\n" + "=" * 50);
    print("LOADING MODELS FOR PREDICTION");
    print("=" * 50)

    class_processor = TaskDataProcessor({ })
    class_processor.load_preprocessors('classification_preprocessors.pkl')
    classifier_checkpoint = torch.load('classification_model.pth' , map_location='cpu')
    classifier = TaskModelFactory.create_classifier(**classifier_checkpoint['model_config'])
    classifier.load_state_dict(classifier_checkpoint['model_state_dict'])
    classifier_trainer = TaskModelFactory.create_trainer(classifier)

    reg_processor = TaskDataProcessor({ })
    reg_processor.load_preprocessors('regression_preprocessors.pkl')
    regressor_checkpoint = torch.load('regression_model.pth' , map_location='cpu')
    regressor = TaskModelFactory.create_regressor(**regressor_checkpoint['model_config'])
    regressor.load_state_dict(regressor_checkpoint['model_state_dict'])
    regressor_trainer = TaskModelFactory.create_trainer(regressor)

    task_scenarios = [
        {
            "task_name":"URGENT: Fix login button" , "description":"Critical bug reported by major client. ASAP." ,
            "creation_date":datetime.now() - timedelta(hours=1) , "due_date":datetime.now() + timedelta(days=1) } ,
        {
            "task_name":"Draft Q4 marketing report" ,
            "description":"Prepare the quarterly report for the upcoming board meeting." ,
            "creation_date":datetime.now() - timedelta(days=2) , "due_date":datetime.now() + timedelta(days=12) ,
            "client_name":"Innovate LLC" , "email":"hello@innovate.com" } ,
        {
            "task_name":"Research new markets" ,
            "description":"Long-term strategic task to identify potential expansion areas." ,
            "creation_date":datetime.now() - timedelta(days=10) , "due_date":datetime.now() + timedelta(days=50) }
    ]
    df_predict = pd.DataFrame(task_scenarios)

    # Pasul 1: Pregătirea datelor pentru predicție
    required_cols = reg_processor.global_averages.keys()
    for col in required_cols:
        if col not in df_predict.columns:
            df_predict[col] = reg_processor.global_averages[col]

    categorical_cols = ['board_name' , 'client_name' , 'email']
    for col in categorical_cols:
        if col not in df_predict.columns:
            df_predict[col] = "Unknown"

    df_predict['days_to_complete'] = (df_predict['due_date'] - df_predict['creation_date']).dt.days
    df_predict['creation_day_of_week'] = df_predict['creation_date'].dt.dayofweek
    df_predict['creation_hour'] = df_predict['creation_date'].dt.hour
    df_predict['creation_month'] = df_predict['creation_date'].dt.month
    df_predict['creation_quarter'] = df_predict['creation_date'].dt.quarter
    df_predict['description_length'] = df_predict['description'].str.len()
    df_predict['task_name_length'] = df_predict['task_name'].str.len()

    # Pasul 2: Procesarea și scalarea separată
    df_class_processed = class_processor.engineer_advanced_features(df_predict.copy() , fit_mode=False)
    class_feature_list = class_processor.feature_names
    df_class_final = df_class_processed[class_feature_list].fillna(0)
    scaled_class_features = class_processor.scaler.transform(df_class_final)

    df_reg_processed = reg_processor.engineer_advanced_features(df_predict.copy() , fit_mode=False)
    reg_feature_list = reg_processor.feature_names
    df_reg_final = df_reg_processed[reg_feature_list].fillna(0)
    scaled_reg_features = reg_processor.scaler.transform(df_reg_final)

    # Pasul 3: Predicția și Analiza SHAP
    print("Predictions and Feedback for new tasks:")
    class_predictions , _ , confidence_scores = classifier_trainer.predict(scaled_class_features ,
                                                                           return_confidence=True)
    time_predictions = regressor_trainer.predict(scaled_reg_features)
    time_predictions = np.maximum(1.0 , time_predictions)

    # Implementarea SHAP
    def regressor_predict_fn(x):
        tensor_x = torch.FloatTensor(x).to(regressor_trainer.device)
        return regressor_trainer.predict(tensor_x)

    background_data_sample = np.random.rand(100 , len(reg_processor.feature_names))
    explainer = shap.KernelExplainer(regressor_predict_fn , background_data_sample)
    shap_values = explainer.shap_values(scaled_reg_features)

    status_labels = { 0:'Pending' , 1:'In Progress' , 2:'Completed' }
    task_descriptions = ["High Priority Short Deadline Task" , "Regular Medium Complexity Task" ,
        "Long-term Project Task (Cold Start)"]

    for i , desc in enumerate(task_descriptions):
        print(f"\n--- {desc} ---")
        print(f"  Predicted Status: {status_labels[class_predictions[i]]} (Confidence: {confidence_scores[i]:.3f})")
        print(f"  Predicted Completion Time: {time_predictions[i]:.2f} days")

        print("\n  Key Factors Influencing Prediction Time:")
        feature_importance = pd.DataFrame(list(zip(reg_processor.feature_names , shap_values[i])) ,
                                          columns=['feature' , 'shap_value'])
        feature_importance['abs_shap'] = feature_importance['shap_value'].abs()
        feature_importance = feature_importance.sort_values(by='abs_shap' , ascending=False)

        for _ , row in feature_importance.head(3).iterrows():
            feature = row['feature']
            value = row['shap_value']
            if value > 0:
                print(f"    - Factor increasing duration: '{feature}' (adds ~{value:.1f} days)")
            elif value < 0:
                print(f"    - Factor decreasing duration: '{feature}' (reduces by ~{-value:.1f} days)")
            else:
                print(f"    - Factor with no impact: '{feature}'")


if __name__ == "__main__":
    print("🚀 Starting ML Pipeline...")
    results = run_pipeline()
    if results:
        try:
            load_and_predict()
        except FileNotFoundError:
            print("Model files not found. Run the main pipeline first.")
        print("✅ TRAINING AND EVALUATION COMPLETED!")
    else:
        print("\n❌ PIPELINE FAILED")