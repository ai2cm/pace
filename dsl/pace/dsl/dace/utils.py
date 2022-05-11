import time

from pace.dsl.dace.dace_config import dace_config


# Rough timer & log for major operations of DaCe build stack
class DaCeProgress:
    def __init__(self, label):
        self.label = label

    @classmethod
    def log(cls, message: str):
        print(f"[{dace_config.get_orchestrate()}] {message}")

    def __enter__(self):
        DaCeProgress.log(f"{self.label}...")
        self.start = time.time()

    def __exit__(self, _type, _val, _traceback):
        elapsed = time.time() - self.start
        DaCeProgress.log(f"{self.label}...{elapsed}s.")
