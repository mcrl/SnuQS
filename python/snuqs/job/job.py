from snuqs.result import Result

class Job:
    def __init__(self, name):
        self.name = name
        self._result = Result(self.name)

    def result(self):
        return self._result
