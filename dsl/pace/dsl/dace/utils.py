import time

from pace.dsl.dace.dace_config import DaceConfig


# Rough timer & log for major operations of DaCe build stack
class DaCeProgress:
    def __init__(self, config: DaceConfig, label: str):
        self.prefix = f"[{config.get_orchestrate()}]"
        self.label = label

    @classmethod
    def log(cls, prefix: str, message: str):
        print(f"{prefix} {message}")

    def __enter__(self):
        DaCeProgress.log(self.prefix, f"{self.label}...")
        self.start = time.time()

    def __exit__(self, _type, _val, _traceback):
        elapsed = time.time() - self.start
        DaCeProgress.log(self.prefix, f"{self.label}...{elapsed}s.")
