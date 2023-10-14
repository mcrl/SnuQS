from snuqs.virtual_device import VirtualDevice
from snuqs.request import Request
from abc import *

class BaseSimulator(VirtualDevice):
    @abstractmethod
    def _run(self, req: Request):
        raise "Cannot be here"
