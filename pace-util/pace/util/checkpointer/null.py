from .base import Checkpointer


class NullCheckpointer(Checkpointer):
    def __call__(self, savepoint_name, **kwargs):
        pass
