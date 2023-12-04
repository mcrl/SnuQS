from snuqs.request import Request
from abc import ABC, abstractmethod


class BaseSimulator(ABC):
    @abstractmethod
    def _run(self, req: Request):
        raise "Cannot be here"
