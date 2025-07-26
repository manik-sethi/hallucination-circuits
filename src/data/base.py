#src/data/base.py

from abc import ABC, abstractmethod


class DatasetLoader(ABC):
    @abstractmethod
    def __len__(self) -> int:
        ...
    
    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        ...