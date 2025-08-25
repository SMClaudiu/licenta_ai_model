#preprocessing

import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder , RobustScaler
from torch.utils.data import WeightedRandomSampler

from src.core.config import AppConfig
from src.data.database import DatabaseManager


class DataPreprocessor:
    def __init__(self):
        self.config = AppConfig()
        self.db = DatabaseManager(self.config.db)

        self.engine = None
        self.label_encoders = { }
        self.frequency_maps = { }
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.feature_names = []
        self.original_feature_names = []
        self.global_averages = { }
        self.quantiles = { }

    def select_best_features(self , X , y , task_type='classification' , k=20):
        if task_type == 'classification':
            selector = SelectKBest(score_func=mutual_info_classif , k=k)
        else:
            selector = SelectKBest(score_func=f_classif , k=k)
        x_selected = selector.fit_transform(X , y)
        selected_indices = selector.get_support(indices=True)
        selected_features = [self.original_feature_names[i] for i in selected_indices]
        print(f"Selected {len(selected_features)} best features:")
        for i , (feature , score) in enumerate(zip(selected_features , selector.scores_[selected_indices])):
            print(f"  {i + 1}. {feature}: {score:.4f}")
        self.feature_selector = selector
        self.feature_names = selected_features
        return x_selected

    def prepare_classification_data(self , df):
        feature_columns = [
            'task_lifecycle_progress' , 'is_newly_created' , 'days_to_complete' , 'creation_day_of_week' ,
            'creation_hour' , 'creation_month' , 'creation_quarter' , 'description_length' , 'task_name_length' ,
            'client_total_tasks' , 'client_completed_tasks' , 'client_pending_tasks' , 'client_avg_task_duration' ,
            'board_total_tasks' , 'board_completed_tasks' , 'board_avg_task_duration' , 'is_overdue' ,
            'is_weekend_created' , 'is_business_hours' , 'is_urgent' , 'is_long_term' , 'creation_hour_sin' ,
            'creation_hour_cos' , 'creation_day_sin' , 'creation_day_cos' , 'creation_month_sin' ,
            'creation_month_cos' ,
            'has_long_description' , 'has_short_description' , 'has_long_task_name' , 'priority_keywords' ,
            'completion_keywords' , 'technical_keywords' , 'client_completion_rate' , 'client_pending_rate' ,
            'client_productivity_score' , 'board_completion_rate' , 'board_efficiency_score' ,
            'task_complexity_vs_client_avg' , 'task_complexity_vs_board_avg' , 'board_name_encoded' ,
            'client_name_encoded' , 'email_encoded' , 'board_name_frequency' , 'client_name_frequency' ,
            'email_frequency' , 'urgency_x_complexity' , 'client_performance_x_task_complexity' ,
            'board_efficiency_x_urgency' , 'weekend_x_urgency' , 'days_since_creation' , 'days_until_due' ,
            'time_pressure'
        ]
        available_features = [col for col in feature_columns if col in df.columns]
        x = df[available_features].fillna(0).values
        self.original_feature_names = available_features
        le_status = LabelEncoder()
        y = le_status.fit_transform(df['status'])
        self.label_encoders['status'] = le_status
        x_selected = self.select_best_features(x , y , 'classification' , k=min(25 , len(available_features)))
        return x_selected , y

    def prepare_regression_data(self , df):
        feature_columns = [
            'creation_day_of_week' , 'creation_hour' , 'creation_month' , 'creation_quarter' ,
            'description_length' , 'task_name_length' , 'client_total_tasks' , 'client_completed_tasks' ,
            'client_pending_tasks' , 'client_avg_task_duration' , 'board_total_tasks' , 'board_completed_tasks' ,
            'board_avg_task_duration' , 'is_weekend_created' , 'is_business_hours' , 'creation_hour_sin' ,
            'creation_hour_cos' , 'creation_day_sin' , 'creation_day_cos' , 'creation_month_sin' ,
            'creation_month_cos' ,
            'has_long_description' , 'has_short_description' , 'has_long_task_name' , 'priority_keywords' ,
            'completion_keywords' , 'technical_keywords' , 'client_completion_rate' , 'client_pending_rate' ,
            'client_productivity_score' , 'board_completion_rate' , 'board_efficiency_score' ,
            'board_name_encoded' , 'client_name_encoded' , 'email_encoded' , 'board_name_frequency' ,
            'client_name_frequency' , 'email_frequency'
        ]
        available_features = [col for col in feature_columns if col in df.columns]
        x = df[available_features].fillna(0).values
        self.original_feature_names = available_features
        y = df['days_to_complete'].fillna(df['days_to_complete'].median()).values
        q1 = np.percentile(y , 25)
        q3 = np.percentile(y , 75)
        IQR = q3 - q1
        lower_bound = q1 - 1.5 * IQR
        upper_bound = q3 + 1.5 * IQR
        mask = (y >= lower_bound) & (y <= upper_bound)
        x = x[mask]
        y = y[mask]
        print(f"Removed {(~mask).sum()} outliers from regression data")
        x_selected = self.select_best_features(x , y , 'regression' , k=min(20 , len(available_features)))
        return x_selected , y

    def get_processed_data(self, task_type='classification'):
        if not self.db.connect_to_db():
            return None
        df = self.db.extract_data()
        if df is None:
            return None
        avg_cols = ['client_total_tasks', 'client_completed_tasks', 'client_pending_tasks',
                    'client_avg_task_duration', 'board_total_tasks', 'board_completed_tasks',
                    'board_avg_task_duration']
        for col in avg_cols:
            self.global_averages[col] = df[col].mean()
        df = self.engineer_features(df, fit_mode=True)
        if task_type == 'classification':
            x, y = self.prepare_classification_data(df)
        else:
            x, y = self.prepare_regression_data(df)
        x_scaled = self.scaler.fit_transform(x)
        return x_scaled, y

    def create_balanced_sampler(self , y):
        class_counts = np.bincount(y)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y]
        return WeightedRandomSampler(weights=sample_weights , num_samples=len(sample_weights) , replacement=True)

    def save(self , filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Preprocessor saved to {filepath}")

    @staticmethod
    def load(filepath: str):
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        print(f"Preprocessor loaded from {filepath}")
        return obj

    def engineer_features(self , df , fit_mode=False):
        df['creation_date'] = pd.to_datetime(df['creation_date'])
        df['due_date'] = pd.to_datetime(df['due_date'])
        current_time = datetime.now()
        df['days_since_creation'] = (current_time - df['creation_date']).dt.days
        df['days_to_complete'] = (df['due_date'] - df['creation_date']).dt.days
        df['task_lifecycle_progress'] = df['days_since_creation'] / (df['days_to_complete'] + 1e-6)
        df['task_lifecycle_progress'] = df['task_lifecycle_progress'].clip(0 , 5)
        df['is_newly_created'] = (df['days_since_creation'] < 1).astype(int)
        df['is_overdue'] = (df['due_date'] < current_time).astype(int)
        df['is_weekend_created'] = df['creation_day_of_week'].isin([0 , 6]).astype(int)
        df['is_business_hours'] = ((df['creation_hour'] >= 9) & (df['creation_hour'] <= 17)).astype(int)
        df['is_urgent'] = (df['days_to_complete'] <= 3).astype(int)
        df['is_long_term'] = (df['days_to_complete'] > 30).astype(int)

        if fit_mode:
            self.quantiles['desc_q25'] = df['description_length'].quantile(0.25)
            self.quantiles['desc_q75'] = df['description_length'].quantile(0.75)
            self.quantiles['name_q75'] = df['task_name_length'].quantile(0.75)

        df['has_long_description'] = (df['description_length'] > self.quantiles.get('desc_q75' , 0)).astype(int)
        df['has_short_description'] = (df['description_length'] < self.quantiles.get('desc_q25' , 0)).astype(int)
        df['has_long_task_name'] = (df['task_name_length'] > self.quantiles.get('name_q75' , 0)).astype(int)

        priority_keywords = ['urgent' , 'asap' , 'priority' , 'critical' , 'important' , 'emergency' , 'immediate']
        completion_keywords = ['review' , 'check' , 'verify' , 'approve' , 'sign' , 'confirm']
        technical_keywords = ['bug' , 'fix' , 'error' , 'debug' , 'test' , 'deploy' , 'update']
        df['priority_keywords'] = df['description'].str.lower().str.contains('|'.join(priority_keywords) ,
                                                                         na=False).astype(int)
        df['completion_keywords'] = df['description'].str.lower().str.contains('|'.join(completion_keywords) ,
                                                                           na=False).astype(int)
        df['technical_keywords'] = df['description'].str.lower().str.contains('|'.join(technical_keywords) ,
                                                                          na=False).astype(int)
        df['client_completion_rate'] = df['client_completed_tasks'] / df['client_total_tasks']
        df['client_pending_rate'] = df['client_pending_tasks'] / df['client_total_tasks']
        df['client_productivity_score'] = df['client_completed_tasks'] / (df['client_avg_task_duration'] + 1)
        df['board_completion_rate'] = df['board_completed_tasks'] / df['board_total_tasks']
        df['board_efficiency_score'] = df['board_completed_tasks'] / (df['board_avg_task_duration'] + 1)
        df['task_complexity_vs_client_avg'] = df['days_to_complete'] / (df['client_avg_task_duration'] + 1)
        df['task_complexity_vs_board_avg'] = df['days_to_complete'] / (df['board_avg_task_duration'] + 1)

        categorical_columns = ['board_name' , 'client_name' , 'email']
        for col in categorical_columns:
            if fit_mode:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                freq_map = df[col].value_counts().to_dict()
                self.frequency_maps[col] = freq_map
                df[f'{col}_frequency'] = df[col].map(freq_map)
            else:
                le = self.label_encoders.get(col)
                if le:
                    known_classes = list(le.classes_)
                    df[f'{col}_encoded'] = df[col].astype(str).apply(
                        lambda x:known_classes.index(x) if x in known_classes else -1)
                freq_map = self.frequency_maps.get(col , { })
                df[f'{col}_frequency'] = df[col].map(freq_map).fillna(0)

        df['urgency_x_complexity'] = df['is_urgent'] * df['days_to_complete']
        df['client_performance_x_task_complexity'] = df['client_completion_rate'] * df['days_to_complete']
        df['board_efficiency_x_urgency'] = df['board_efficiency_score'] * df['is_urgent']
        df['weekend_x_urgency'] = df['is_weekend_created'] * df['is_urgent']
        df['days_until_due'] = (df['due_date'] - current_time).dt.days
        df['time_pressure'] = np.where(df['days_until_due'] > 0 , df['days_since_creation'] / df['days_until_due'] , 10)
        return df

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('db' , None)
        state.pop('engine' , None)
        return state

    def __setstate__(self , state):
        self.__dict__.update(state)
        self.db = None
        self.engine = None
