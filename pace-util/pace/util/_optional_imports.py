class RaiseWhenAccessed:
    def __init__(self, err):
        self._err = err

    def __getattr__(self, _):
        raise self._err

    def __call__(self, *args, **kwargs):
        raise self._err
