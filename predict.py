import torch
import numpy as np
import pandas as pd
from datetime import datetime , timedelta
import shap

from src.data.preprocessing import DataPreprocessor
from src.models.factory import TaskModelFactory


def get_predictions_and_feedback():
    print("\n" + "=" * 50)
    print("LOADING MODELS FOR PREDICTION")
    print("=" * 50)

    try:
        class_processor = DataPreprocessor.load('artifacts/classification_preprocessors.pkl')

        class_checkpoint = torch.load('artifacts/classification_model.pth' , map_location='cpu' , weights_only=True)
        classifier = TaskModelFactory.create_classifier(**class_checkpoint['model_config'])
        classifier.load_state_dict(class_checkpoint['model_state_dict'])
        classifier_trainer = TaskModelFactory.create_trainer(classifier)

        reg_processor = DataPreprocessor.load('artifacts/regression_preprocessors.pkl')
        reg_checkpoint = torch.load('artifacts/regression_model.pth' , map_location='cpu' , weights_only=True)
        regressor = TaskModelFactory.create_regressor(**reg_checkpoint['model_config'])
        regressor.load_state_dict(reg_checkpoint['model_state_dict'])
        regressor_trainer = TaskModelFactory.create_trainer(regressor)
    except FileNotFoundError as e:
        print(f"Error: Could not find model artifact - {e}")
        print("Please run main.py to train and save the models first.")
        return

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
            "task_name":"Research new markets" , "description":"Long-term strategic task." ,
            "creation_date":datetime.now() - timedelta(days=10) , "due_date":datetime.now() + timedelta(days=50) }
    ]
    df_predict = pd.DataFrame(task_scenarios)

    # Pregătirea datelor pentru predicție
    for col , avg_val in reg_processor.global_averages.items():
        if col not in df_predict.columns:
            df_predict[col] = avg_val
    for col in ['board_name' , 'client_name' , 'email']:
        if col not in df_predict.columns:
            df_predict[col] = "Unknown"

    df_predict['days_to_complete'] = (
                pd.to_datetime(df_predict['due_date']) - pd.to_datetime(df_predict['creation_date'])).dt.days
    df_predict['creation_day_of_week'] = pd.to_datetime(df_predict['creation_date']).dt.dayofweek
    df_predict['creation_hour'] = pd.to_datetime(df_predict['creation_date']).dt.hour
    df_predict['creation_month'] = pd.to_datetime(df_predict['creation_date']).dt.month
    df_predict['creation_quarter'] = pd.to_datetime(df_predict['creation_date']).dt.quarter
    df_predict['description_length'] = df_predict['description'].str.len()
    df_predict['task_name_length'] = df_predict['task_name'].str.len()

    # Procesare și scalare
    df_class_processed = class_processor.engineer_features(df_predict.copy() , fit_mode=False)
    df_class_final = df_class_processed[class_processor.feature_names].fillna(0)
    scaled_class_features = class_processor.scaler.transform(df_class_final.values)

    df_reg_processed = reg_processor.engineer_features(df_predict.copy() , fit_mode=False)
    df_reg_final = df_reg_processed[reg_processor.feature_names].fillna(0)
    scaled_reg_features = reg_processor.scaler.transform(df_reg_final.values)

    # Predicție și Analiză SHAP
    print("Predictions and Feedback for new tasks:")
    class_predictions , _ , confidence_scores = classifier_trainer.predict(scaled_class_features ,
                                                                           return_confidence=True)
    time_predictions = regressor_trainer.predict(scaled_reg_features)
    time_predictions = np.maximum(1.0 , time_predictions)

    def regressor_predict_fn(x):
        return regressor_trainer.predict(x)

    background_data_sample = np.random.rand(100 , len(reg_processor.feature_names))
    explainer = shap.KernelExplainer(regressor_predict_fn , background_data_sample)
    shap_values = explainer.shap_values(scaled_reg_features)

    status_labels = { 0:'Pending' , 1:'In Progress' , 2:'Completed' }
    task_descriptions = ["High Priority Task" , "Regular Task" , "Cold Start Task"]

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
            feature , value = row['feature'] , row['shap_value']
            if value > 0.1:
                print(f"    - Factor increasing duration: '{feature}' (adds ~{value:.1f} days)")
            elif value < -0.1:
                print(f"    - Factor decreasing duration: '{feature}' (reduces by ~{-value:.1f} days)")


if __name__ == "__main__":
    get_predictions_and_feedback()