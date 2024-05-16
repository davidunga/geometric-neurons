from paths import KEYS_DIR
import random
from time import sleep


def get_key(name: str):
    n_tries = 10
    max_trying_seconds = 1
    max_wait_seconds = float(max_trying_seconds) / n_tries
    for try_count in range(n_tries):
        try:
            with (KEYS_DIR / name).open('r') as f:
                key = f.read().strip()
        except PermissionError:
            if try_count == n_tries - 1:
                raise
            sleep(random.random() * max_wait_seconds)
    return key
