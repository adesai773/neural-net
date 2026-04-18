from abc import ABC, abstractmethod


class LossFunction(ABC):
    @abstractmethod
    def compute_loss(self):
        pass

    @abstractmethod
    def __str__(self) -> str:
        return "LossFunction()"


class MseLoss(LossFunction):
    def compute_loss(self):
        pass

    def __str__(self) -> str:
        return "MseLoss()"


class BceLoss(LossFunction):
    def compute_loss(self):
        pass

    def __str__(self) -> str:
        return "BceLoss()"
