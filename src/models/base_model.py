from abc import ABC , abstractmethod
import torch
import torch.nn as nn
from typing import Dict , Any


class BaseTaskModel(nn.Module , ABC):
    def __init__(self , input_size: int , **kwargs):
        super().__init__()
        self.input_size = input_size

    @abstractmethod
    def forward(self , x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def predict_proba(self , x: torch.Tensor) -> torch.Tensor:
        pass

    def get_model_info(self) -> Dict[str , Any]:
        return {
            'input_size':self.input_size ,
            'num_parameters':sum(p.numel() for p in self.parameters()) ,
            'model_type':self.__class__.__name__
        }