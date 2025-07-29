"""
Enhanced Data Processing and Feature Engineering
Key improvements for better data quality and model performance
"""
import pickle
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sqlalchemy import create_engine
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import warnings

warnings.filterwarnings('ignore')


class EnhancedTaskDataProcessor:
    """Enhanced data processor with better feature engineering"""

    def __init__(self, db_config):
        self.db_config = db_config
        self.engine = None
        self.label_encoders = {}
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_selector = None
        self.feature_names = []
        self.original_feature_names = []

    def connect_to_db(self):
        """Create database connection"""
        try:
            connection_string = f"postgresql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            self.engine = create_engine(connection_string)
            print("Database connection established successfully")
            return True
        except Exception as e:
            print(f"Database connection failed: {e}")
            return False

    def extract_data(self):
        """Enhanced data extraction with more features"""
        query = """
                SELECT t.task_id, \
                       t.creation_date, \
                       t.description, \
                       t.due_date, \
                       t.name                                           as task_name, \
                       t.status, \
                       t.board_id, \
                       b.name                                           as board_name, \
                       b.dash_board_id, \
                       c.client_id, \
                       c.email, \
                       c.name                                           as client_name, \
                       c.phone_number, \
                       EXTRACT(EPOCH FROM (t.due_date::timestamp - t.creation_date::timestamp))::float / 86400 as days_to_complete, EXTRACT(DOW FROM t.creation_date::timestamp) as creation_day_of_week, \
                       EXTRACT(HOUR FROM t.creation_date::timestamp)    as creation_hour, \
                       EXTRACT(MONTH FROM t.creation_date::timestamp)   as creation_month, \
                       EXTRACT(QUARTER FROM t.creation_date::timestamp) as creation_quarter, \
                       LENGTH(t.description)                            as description_length, \
                       LENGTH(t.name)                                   as task_name_length, \
                       COUNT(*)                                            OVER (PARTITION BY c.client_id) as client_total_tasks, COUNT(*) FILTER (WHERE t.status = 2) OVER (PARTITION BY c.client_id) as client_completed_tasks, COUNT(*) FILTER (WHERE t.status = 0) OVER (PARTITION BY c.client_id) as client_pending_tasks, AVG(EXTRACT(EPOCH FROM (t.due_date::timestamp - t.creation_date::timestamp))::float / 86400) \
                    OVER (PARTITION BY c.client_id) as client_avg_task_duration, COUNT(*) OVER (PARTITION BY t.board_id) as board_total_tasks, COUNT(*) FILTER (WHERE t.status = 2) OVER (PARTITION BY t.board_id) as board_completed_tasks, AVG(EXTRACT(EPOCH FROM (t.due_date::timestamp - t.creation_date::timestamp))::float / 86400) \
                    OVER (PARTITION BY t.board_id) as board_avg_task_duration
                FROM task t
                         JOIN board b ON t.board_id = b.board_id
                         JOIN dashboard d ON b.dash_board_id = d.id
                         JOIN client c ON d.client_id = c.client_id
                WHERE t.due_date IS NOT NULL
                  AND t.creation_date IS NOT NULL
                  AND EXTRACT(EPOCH FROM (t.due_date::timestamp - t.creation_date::timestamp)) > 0
                ORDER BY t.creation_date \
                """

        try:
            df = pd.read_sql_query(query, self.engine)
            print(f"Extracted {len(df)} records from database")

            # Remove extreme outliers (tasks longer than 1 year)
            df = df[df['days_to_complete'] <= 365]
            print(f"After filtering outliers: {len(df)} records")

            return df
        except Exception as e:
            print(f"Data extraction failed: {e}")
            return None

    def engineer_advanced_features(self, df):
        """Enhanced feature engineering with more sophisticated features"""

        # Convert datetime columns
        df['creation_date'] = pd.to_datetime(df['creation_date'])
        df['due_date'] = pd.to_datetime(df['due_date'])
        current_time = datetime.now()

        # === TIME-BASED FEATURES ===
        df['is_overdue'] = (df['due_date'] < current_time).astype(int)
        df['is_weekend_created'] = df['creation_day_of_week'].isin([0, 6]).astype(int)
        df['is_business_hours'] = ((df['creation_hour'] >= 9) & (df['creation_hour'] <= 17)).astype(int)
        df['is_urgent'] = (df['days_to_complete'] <= 3).astype(int)
        df['is_long_term'] = (df['days_to_complete'] > 30).astype(int)

        # Cyclical encoding for time features
        df['creation_hour_sin'] = np.sin(2 * np.pi * df['creation_hour'] / 24)
        df['creation_hour_cos'] = np.cos(2 * np.pi * df['creation_hour'] / 24)
        df['creation_day_sin'] = np.sin(2 * np.pi * df['creation_day_of_week'] / 7)
        df['creation_day_cos'] = np.cos(2 * np.pi * df['creation_day_of_week'] / 7)
        df['creation_month_sin'] = np.sin(2 * np.pi * df['creation_month'] / 12)
        df['creation_month_cos'] = np.cos(2 * np.pi * df['creation_month'] / 12)

        # === TEXT-BASED FEATURES ===
        df['has_long_description'] = (df['description_length'] > df['description_length'].quantile(0.75)).astype(int)
        df['has_short_description'] = (df['description_length'] < df['description_length'].quantile(0.25)).astype(int)
        df['has_long_task_name'] = (df['task_name_length'] > df['task_name_length'].quantile(0.75)).astype(int)

        # Priority and urgency keywords
        priority_keywords = ['urgent', 'asap', 'priority', 'critical', 'important', 'emergency', 'immediate']
        completion_keywords = ['review', 'check', 'verify', 'approve', 'sign', 'confirm']
        technical_keywords = ['bug', 'fix', 'error', 'debug', 'test', 'deploy', 'update']

        df['priority_keywords'] = df['description'].str.lower().str.contains(
            '|'.join(priority_keywords), na=False
        ).astype(int)
        df['completion_keywords'] = df['description'].str.lower().str.contains(
            '|'.join(completion_keywords), na=False
        ).astype(int)
        df['technical_keywords'] = df['description'].str.lower().str.contains(
            '|'.join(technical_keywords), na=False
        ).astype(int)

        # === WORKLOAD AND PERFORMANCE FEATURES ===
        # Client performance metrics
        df['client_completion_rate'] = df['client_completed_tasks'] / df['client_total_tasks']
        df['client_pending_rate'] = df['client_pending_tasks'] / df['client_total_tasks']
        df['client_productivity_score'] = df['client_completed_tasks'] / (df['client_avg_task_duration'] + 1)

        # Board performance metrics
        df['board_completion_rate'] = df['board_completed_tasks'] / df['board_total_tasks']
        df['board_efficiency_score'] = df['board_completed_tasks'] / (df['board_avg_task_duration'] + 1)

        # Relative complexity scores
        df['task_complexity_vs_client_avg'] = df['days_to_complete'] / (df['client_avg_task_duration'] + 1)
        df['task_complexity_vs_board_avg'] = df['days_to_complete'] / (df['board_avg_task_duration'] + 1)

        # === CATEGORICAL ENCODINGS ===
        # Enhanced categorical encoding with frequency
        categorical_columns = ['board_name', 'client_name', 'email']
        for col in categorical_columns:
            # Label encoding
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

            # Frequency encoding
            freq_map = df[col].value_counts().to_dict()
            df[f'{col}_frequency'] = df[col].map(freq_map)

        # === INTERACTION FEATURES ===
        # Create interaction features between important variables
        df['urgency_x_complexity'] = df['is_urgent'] * df['days_to_complete']
        df['client_performance_x_task_complexity'] = df['client_completion_rate'] * df['days_to_complete']
        df['board_efficiency_x_urgency'] = df['board_efficiency_score'] * df['is_urgent']
        df['weekend_x_urgency'] = df['is_weekend_created'] * df['is_urgent']

        # === DERIVED TIME FEATURES ===
        df['days_since_creation'] = (current_time - df['creation_date']).dt.days
        df['days_until_due'] = (df['due_date'] - current_time).dt.days
        df['time_pressure'] = np.where(df['days_until_due'] > 0,
                                       df['days_since_creation'] / df['days_until_due'],
                                       10)  # High pressure if overdue

        return df

    def select_best_features(self, X, y, task_type='classification', k=20):
        """Select the best features using statistical tests"""

        if task_type == 'classification':
            # Use mutual information for classification
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        else:
            # Use F-statistic for regression
            selector = SelectKBest(score_func=f_classif, k=k)

        X_selected = selector.fit_transform(X, y)

        # Get selected feature indices
        selected_indices = selector.get_support(indices=True)
        selected_features = [self.original_feature_names[i] for i in selected_indices]

        print(f"Selected {len(selected_features)} best features:")
        for i, (feature, score) in enumerate(zip(selected_features, selector.scores_[selected_indices])):
            print(f"  {i + 1}. {feature}: {score:.4f}")

        self.feature_selector = selector
        self.feature_names = selected_features

        return X_selected

    def prepare_classification_data(self, df):
        """Enhanced classification data preparation"""

        # Define all possible features
        feature_columns = [
            'days_to_complete', 'creation_day_of_week', 'creation_hour', 'creation_month', 'creation_quarter',
            'description_length', 'task_name_length', 'client_total_tasks', 'client_completed_tasks',
            'client_pending_tasks', 'client_avg_task_duration', 'board_total_tasks', 'board_completed_tasks',
            'board_avg_task_duration', 'is_overdue', 'is_weekend_created', 'is_business_hours', 'is_urgent',
            'is_long_term', 'creation_hour_sin', 'creation_hour_cos', 'creation_day_sin', 'creation_day_cos',
            'creation_month_sin', 'creation_month_cos', 'has_long_description', 'has_short_description',
            'has_long_task_name', 'priority_keywords', 'completion_keywords', 'technical_keywords',
            'client_completion_rate', 'client_pending_rate', 'client_productivity_score', 'board_completion_rate',
            'board_efficiency_score', 'task_complexity_vs_client_avg', 'task_complexity_vs_board_avg',
            'board_name_encoded', 'client_name_encoded', 'email_encoded', 'board_name_frequency',
            'client_name_frequency', 'email_frequency', 'urgency_x_complexity', 'client_performance_x_task_complexity',
            'board_efficiency_x_urgency', 'weekend_x_urgency', 'days_since_creation', 'days_until_due', 'time_pressure'
        ]

        # Filter only existing columns and handle missing values
        available_features = [col for col in feature_columns if col in df.columns]
        X = df[available_features].fillna(0).values
        self.original_feature_names = available_features

        # Encode status labels
        le_status = LabelEncoder()
        y = le_status.fit_transform(df['status'])
        self.label_encoders['status'] = le_status

        # Feature selection
        X_selected = self.select_best_features(X, y, 'classification', k=min(25, len(available_features)))

        return X_selected, y

    def prepare_regression_data(self, df):
        """Enhanced regression data preparation"""

        # Same feature engineering as classification
        feature_columns = [
            'creation_day_of_week', 'creation_hour', 'creation_month', 'creation_quarter',
            'description_length', 'task_name_length', 'client_total_tasks', 'client_completed_tasks',
            'client_pending_tasks', 'client_avg_task_duration', 'board_total_tasks', 'board_completed_tasks',
            'board_avg_task_duration', 'is_overdue', 'is_weekend_created', 'is_business_hours', 'is_urgent',
            'is_long_term', 'creation_hour_sin', 'creation_hour_cos', 'creation_day_sin', 'creation_day_cos',
            'creation_month_sin', 'creation_month_cos', 'has_long_description', 'has_short_description',
            'has_long_task_name', 'priority_keywords', 'completion_keywords', 'technical_keywords',
            'client_completion_rate', 'client_pending_rate', 'client_productivity_score', 'board_completion_rate',
            'board_efficiency_score', 'board_name_encoded', 'client_name_encoded', 'email_encoded',
            'board_name_frequency', 'client_name_frequency', 'email_frequency', 'days_since_creation',
            'days_until_due', 'time_pressure'
        ]

        available_features = [col for col in feature_columns if col in df.columns]
        X = df[available_features].fillna(0).values
        self.original_feature_names = available_features

        # Target variable with outlier handling
        y = df['days_to_complete'].fillna(df['days_to_complete'].median()).values

        # Remove extreme outliers using IQR method
        Q1 = np.percentile(y, 25)
        Q3 = np.percentile(y, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Keep only samples within bounds
        mask = (y >= lower_bound) & (y <= upper_bound)
        X = X[mask]
        y = y[mask]

        print(f"Removed {(~mask).sum()} outliers from regression data")

        # Feature selection
        X_selected = self.select_best_features(X, y, 'regression', k=min(25, len(available_features)))

        return X_selected, y

    def get_processed_data(self, task_type='classification'):
        """Get fully processed data with enhanced features"""
        if not self.connect_to_db():
            return None

        df = self.extract_data()
        if df is None:
            return None

        df = self.engineer_advanced_features(df)

        if task_type == 'classification':
            X, y = self.prepare_classification_data(df)
        else:
            X, y = self.prepare_regression_data(df)

        # Scale features using RobustScaler (better for outliers)
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y

    def create_balanced_sampler(self, y):
        """Create a weighted sampler for balanced training"""
        class_counts = np.bincount(y)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y]

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

    def save_preprocessors(self, filepath='enhanced_preprocessors.pkl'):
        """Save all preprocessing components"""
        preprocessors = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'original_feature_names': self.original_feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessors, f)
        print(f"Enhanced preprocessors saved to {filepath}")

    def load_preprocessors(self, filepath='enhanced_preprocessors.pkl'):
        """Load saved preprocessors"""
        with open(filepath, 'rb') as f:
            preprocessors = pickle.load(f)
        self.label_encoders = preprocessors['label_encoders']
        self.scaler = preprocessors['scaler']
        self.feature_selector = preprocessors.get('feature_selector', None)
        self.feature_names = preprocessors['feature_names']
        self.original_feature_names = preprocessors.get('original_feature_names', self.feature_names)
        print(f"Enhanced preprocessors loaded from {filepath}")


class EnhancedTaskDatasetFactory:
    """Enhanced factory with better data handling"""

    def __init__(self, data_processor):
        self.data_processor = data_processor

    def create_classification_datasets(self, test_size=0.2, val_size=0.1, batch_size=64, use_stratification=True):
        """Create enhanced classification datasets with better sampling"""
        X, y = self.data_processor.get_processed_data('classification')
        if X is None:
            return None

        return self._create_enhanced_datasets(X, y, test_size, val_size, batch_size, 'classification',
                                              use_stratification)

    def create_regression_datasets(self, test_size=0.2, val_size=0.1, batch_size=64):
        """Create enhanced regression datasets"""
        X, y = self.data_processor.get_processed_data('regression')
        if X is None:
            return None

        return self._create_enhanced_datasets(X, y, test_size, val_size, batch_size, 'regression', False)

    def _create_enhanced_datasets(self, X, y, test_size, val_size, batch_size, task_type, use_stratification):
        """Internal method with enhanced dataset creation"""

        # Enhanced data splitting
        if task_type == 'classification' and use_stratification:
            # Use stratified split for classification
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size / (1 - test_size), random_state=42, stratify=y_temp
            )
        else:
            # Regular split for regression
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size / (1 - test_size), random_state=42
            )

        # Create PyTorch datasets
        train_dataset = TaskDataset(X_train, y_train)
        val_dataset = TaskDataset(X_val, y_val)
        test_dataset = TaskDataset(X_test, y_test)

        # Enhanced DataLoaders
        if task_type == 'classification':
            # Use balanced sampling for training
            sampler = self.data_processor.create_balanced_sampler(y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Print enhanced statistics
        print(f"Enhanced {task_type.capitalize()} dataset created:")
        print(f"Train: {len(train_dataset)} samples")
        print(f"Validation: {len(val_dataset)} samples")
        print(f"Test: {len(test_dataset)} samples")
        print(f"Selected features: {X_train.shape[1]}")

        if task_type == 'classification':
            unique, counts = np.unique(y_train, return_counts=True)
            print(f"Class distribution in training set:")
            for class_idx, count in zip(unique, counts):
                print(f"  Class {class_idx}: {count} samples ({count / len(y_train) * 100:.1f}%)")

        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'feature_names': self.data_processor.feature_names,
            'num_features': X_train.shape[1],
            'num_classes': len(np.unique(y)) if task_type == 'classification' else 1,
            'task_type': task_type,
            'class_distribution': dict(
                zip(*np.unique(y_train, return_counts=True))) if task_type == 'classification' else None
        }


class TaskDataset(Dataset):
    """Enhanced PyTorch Dataset with data augmentation options"""

    def __init__(self, features, targets=None, transform=None, add_noise=False, noise_factor=0.01):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        self.transform = transform
        self.add_noise = add_noise
        self.noise_factor = noise_factor

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = self.features[idx]

        # Optional: Add small amount of noise for regularization
        if self.add_noise and self.training:
            noise = torch.randn_like(sample) * self.noise_factor
            sample = sample + noise

        if self.transform:
            sample = self.transform(sample)

        if self.targets is not None:
            return sample, self.targets[idx]
        return sample