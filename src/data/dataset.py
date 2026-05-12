#dataset

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from src.data.preprocessing import DataPreprocessor

class DatasetFactory:
    def __init__(self , data_preprocessor: DataPreprocessor):
        self.data_preprocessor = data_preprocessor

    def create_classification_datasets(self , test_size=0.2 , val_size=0.1 , batch_size=64 , use_stratification=True):
        x , y = self.data_preprocessor.get_processed_data('classification')
        if x is None:
            return None
        return self.create_datasets(x , y , test_size , val_size , batch_size , 'classification' , use_stratification)

    def create_regression_datasets(self , test_size=0.2 , val_size=0.1 , batch_size=64):
        x , y = self.data_preprocessor.get_processed_data('regression')
        if x is None:
            return None
        return self.create_datasets(x , y , test_size , val_size , batch_size , 'regression' , False)

    def create_datasets(self , x , y , test_size , val_size , batch_size , task_type , use_stratification):
        if task_type == 'classification' and use_stratification:
            x_temp , x_test , y_temp , y_test = train_test_split(x , y , test_size=test_size , random_state=42 ,
                                                                 stratify=y)
            x_train , x_val , y_train , y_val = train_test_split(x_temp , y_temp ,
                                                                 test_size=val_size / (1 - test_size) ,
                                                                 random_state=42 , stratify=y_temp)
        else:
            x_temp , x_test , y_temp , y_test = train_test_split(x , y , test_size=test_size , random_state=42)
            x_train , x_val , y_train , y_val = train_test_split(x_temp , y_temp ,
                                                                 test_size=val_size / (1 - test_size) , random_state=42)
        train_dataset = TaskDataset(x_train , y_train , add_noise=True , noise_factor=0.01)
        val_dataset = TaskDataset(x_val , y_val , add_noise=False)
        test_dataset = TaskDataset(x_test , y_test , add_noise=False)
        if task_type == 'classification':
            sampler = self.data_preprocessor.create_balanced_sampler(y_train)
            train_loader = DataLoader(train_dataset , batch_size=batch_size , sampler=sampler)
        else:
            train_loader = DataLoader(train_dataset , batch_size=batch_size , shuffle=True)
        val_loader = DataLoader(val_dataset , batch_size=batch_size , shuffle=False)
        test_loader = DataLoader(test_dataset , batch_size=batch_size , shuffle=False)
        print(f"{task_type.capitalize()} dataset created:")
        print(f"Train: {len(train_dataset)} samples")
        print(f"Validation: {len(val_dataset)} samples")
        print(f"Test: {len(test_dataset)} samples")
        if task_type == 'classification':
            unique , counts = np.unique(y_train , return_counts=True)
            print(f"Class distribution in training set:")
            for class_idx , count in zip(unique , counts):
                print(f"  Class {class_idx}: {count} samples ({count / len(y_train) * 100:.1f}%)")
        return {
            'train_loader':train_loader , 'val_loader':val_loader , 'test_loader':test_loader ,
            'feature_names':self.data_preprocessor.feature_names , 'num_features':x_train.shape[1] ,
            'num_classes':len(np.unique(y)) if task_type == 'classification' else 1 ,
            'task_type':task_type ,
            'class_distribution':dict(
                zip(*np.unique(y_train , return_counts=True))) if task_type == 'classification' else None
        }

class   TaskDataset(Dataset):
    def __init__(self , features , targets=None , transform=None , add_noise=False , noise_factor=0.01):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        self.transform = transform
        self.add_noise = add_noise
        self.noise_factor = noise_factor

    def __len__(self):
        return len(self.features)

    def __getitem__(self , idx):
        sample = self.features[idx]
        if self.add_noise:
            noise = torch.randn_like(sample) * self.noise_factor
            sample = sample + noise
        if self.transform:
            sample = self.transform(sample)
        if self.targets is not None:
            return sample , self.targets[idx]
        return sample