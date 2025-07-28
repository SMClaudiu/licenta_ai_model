import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sqlalchemy import create_engine
from torch.utils.data import Dataset, DataLoader


class TaskDataset(Dataset):
    """Custom PyTorch Dataset for task management data"""

    def __init__(self, features, targets=None, transform=None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = self.features[idx]
        if self.transform:
            sample = self.transform(sample)

        if self.targets is not None:
            return sample, self.targets[idx]
        return sample


class TaskDataProcessor:
    """Handles data extraction, feature engineering, and preprocessing"""

    def __init__(self, db_config):
        self.db_config = db_config
        self.engine = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = [
            'days_to_complete', 'creation_day_of_week', 'creation_hour',
            'description_length', 'client_task_count', 'board_task_count',
            'is_overdue', 'is_weekend_created', 'is_urgent', 'has_long_description',
            'priority_keywords', 'board_name_encoded', 'client_name_encoded', 'email_encoded'
        ]

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
        """Extract data from all tables with proper joins"""
        query = """
                SELECT t.task_id,
                       t.creation_date,
                       t.description,
                       t.due_date,
                       t.name                                        as task_name,
                       t.status,
                       t.board_id,
                       b.name                                        as board_name,
                       b.dash_board_id,
                       c.client_id,
                       c.email,
                       c.name                                        as client_name,
                       c.phone_number,
                       EXTRACT(EPOCH FROM (t.due_date::timestamp - t.creation_date::timestamp))::float / 86400 as days_to_complete, EXTRACT(DOW FROM t.creation_date::timestamp) as creation_day_of_week,
                       EXTRACT(HOUR FROM t.creation_date::timestamp) as creation_hour,
                       LENGTH(t.description)                         as description_length,
                       COUNT(*)                                         OVER (PARTITION BY c.client_id) as client_task_count, COUNT(*) OVER (PARTITION BY t.board_id) as board_task_count
                FROM task t
                         JOIN board b ON t.board_id = b.board_id
                         JOIN dashboard d ON b.dash_board_id = d.id
                         JOIN client c ON d.client_id = c.client_id
                ORDER BY t.creation_date; \
                """

        try:
            df = pd.read_sql_query(query, self.engine)
            print(f"Extracted {len(df)} records from database")
            return df
        except Exception as e:
            print(f"Data extraction failed: {e}")
            return None

    def engineer_features(self, df):
        """Create additional features from the raw data"""
        # Time-based features
        df['creation_date'] = pd.to_datetime(df['creation_date'])
        df['due_date'] = pd.to_datetime(df['due_date'])

        # Feature engineering
        df['is_overdue'] = (df['due_date'] < datetime.now()).astype(int)
        df['is_weekend_created'] = df['creation_day_of_week'].isin([0, 6]).astype(int)
        df['is_urgent'] = (df['days_to_complete'] <= 7).astype(int)

        # Text features (basic)
        df['has_long_description'] = (df['description_length'] > 100).astype(int)
        df['priority_keywords'] = df['description'].str.contains('urgent|asap|priority|critical', case=False,
                                                                 na=False).astype(int)

        # Categorical encodings
        categorical_columns = ['board_name', 'client_name', 'email']
        for col in categorical_columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        return df

    def prepare_classification_data(self, df):
        """Prepare data for task status classification"""
        X = df[self.feature_names].fillna(0).values

        # Encode status labels
        le_status = LabelEncoder()
        y = le_status.fit_transform(df['status'])
        self.label_encoders['status'] = le_status

        return X, y

    def prepare_regression_data(self, df):
        """Prepare data for completion time regression"""
        X = df[self.feature_names].fillna(0).values
        y = df['days_to_complete'].fillna(df['days_to_complete'].mean()).values

        return X, y

    def get_processed_data(self, task_type='classification'):
        """Get fully processed data for specified task type"""
        if not self.connect_to_db():
            return None

        df = self.extract_data()
        if df is None:
            return None

        df = self.engineer_features(df)

        if task_type == 'classification':
            X, y = self.prepare_classification_data(df)
        else:
            X, y = self.prepare_regression_data(df)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y

    def save_preprocessors(self, filepath='preprocessors.pkl'):
        """Save label encoders and scaler for future use"""
        preprocessors = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessors, f)
        print(f"Preprocessors saved to {filepath}")

    def load_preprocessors(self, filepath='preprocessors.pkl'):
        """Load saved preprocessors"""
        with open(filepath, 'rb') as f:
            preprocessors = pickle.load(f)
        self.label_encoders = preprocessors['label_encoders']
        self.scaler = preprocessors['scaler']
        self.feature_names = preprocessors['feature_names']
        print(f"Preprocessors loaded from {filepath}")


class TaskDatasetFactory:
    """Factory class to create datasets for different ML tasks"""

    def __init__(self, data_processor):
        self.data_processor = data_processor

    def create_classification_datasets(self, test_size=0.2, val_size=0.1, batch_size=32):
        """Create datasets for task status classification"""
        X, y = self.data_processor.get_processed_data('classification')
        if X is None:
            return None

        return self._create_datasets(X, y, test_size, val_size, batch_size, 'classification')

    def create_regression_datasets(self, test_size=0.2, val_size=0.1, batch_size=32):
        """Create datasets for completion time regression"""
        X, y = self.data_processor.get_processed_data('regression')
        if X is None:
            return None

        return self._create_datasets(X, y, test_size, val_size, batch_size, 'regression')

    def _create_datasets(self, X, y, test_size, val_size, batch_size, task_type):
        """Internal method to create train/val/test splits"""
        # Split data
        if task_type == 'classification':
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size / (1 - test_size), random_state=42, stratify=y_temp
            )
        else:
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

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"{task_type.capitalize()} dataset created successfully:")
        print(f"Train: {len(train_dataset)} samples")
        print(f"Validation: {len(val_dataset)} samples")
        print(f"Test: {len(test_dataset)} samples")
        print(f"Feature dimensions: {X_train.shape[1]}")

        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'feature_names': self.data_processor.feature_names,
            'num_features': X_train.shape[1],
            'num_classes': len(np.unique(y)) if task_type == 'classification' else 1,
            'task_type': task_type
        }