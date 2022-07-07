import abc


class Checkpointer(abc.ABC):
    @abc.abstractmethod
    def __call__(self, savepoint_name, **kwargs):
        ...
