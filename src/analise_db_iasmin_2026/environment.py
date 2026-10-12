import os
from abc import ABC, abstractmethod
from pathlib import Path


class Environment(ABC):
    @abstractmethod
    def is_kaggle(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_colab(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_laptop(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_file_store_path(self) -> Path:
        raise NotImplementedError

    @staticmethod
    def get_environment() -> "Environment":
        if os.path.isdir("/kaggle/working"):
            return TupleEnvironment(True, False, False)
        try:
            if os.uname().nodename == "zazazaza":
                return TupleEnvironment(False, False, True)
        except Exception:
            pass
        if os.path.isdir("/kaggle") or os.path.isdir("/content"):
            return TupleEnvironment(False, True, False)
        return TupleEnvironment(False, False, False)


class TupleEnvironment(Environment):
    def __init__(self, is_kaggle: bool, is_colab: bool, is_laptop: bool) -> None:
        super().__init__()
        self._is_kaggle = is_kaggle
        self._is_colab = is_colab
        self._is_laptop = is_laptop

    def is_kaggle(self) -> bool:
        return self._is_kaggle

    def is_colab(self) -> bool:
        return self._is_colab

    def is_laptop(self) -> bool:
        return self._is_laptop

    def get_file_store_path(self) -> Path:
        if self.is_laptop():
            return Path("/mnt/sghdd/adbiasminfilestore/")
        elif self.is_colab():
            return Path("/content/drive/MyDrive/filestore")
        elif self.is_kaggle():
            return Path("/kaggle/working/filestore")
        else:
            return Path("./filestore").absolute()
