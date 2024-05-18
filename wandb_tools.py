import os
import subprocess
from glob import glob
import json
from pathlib import Path
from typing import Sequence
import paths

STATES = 'error', 'failed', 'crashed', 'running', 'finished'
VALID_STATES = 'running', 'finished'


def is_valid_wandb_state(state: str) -> bool:
    assert state in STATES, state
    return state in VALID_STATES


def sync_wandb_run(run_path: str, only_non_synced: bool = False):
    cmnd = ['wandb', 'sync', run_path]
    if only_non_synced:
        cmnd.append('--no-include-synced')
    try:
        subprocess.run(cmnd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError:
        raise


class WandbMgr:

    _run_prefix = 'run-'

    def __init__(self, root: Path | str = paths.WANDB_ROOT):
        self._root = Path(root) / 'wandb'

    def sync_all(self, only_non_synced: bool = True):
        run_path = str(self._root / (self._run_prefix + '*'))
        sync_wandb_run(run_path, only_non_synced=only_non_synced)

    def get_run_names(self, wild: str = '*', state: str | Sequence[str] = None) -> list[str]:
        if not wild.startswith(self._run_prefix):
            wild = f"{self._run_prefix}*{wild}".replace("**", "*")
        run_names = [os.path.basename(d) for d in glob(str(self._root / wild))]
        if state is not None:
            if isinstance(state, str):
                if state == 'valid':
                    state = VALID_STATES
                elif state == 'invalid':
                    state = [s for s in STATES if s not in VALID_STATES]
                else:
                    state = [state]
            assert all(s in STATES for s in state)
            run_names = [run_name for run_name in run_names if self.get_state(run_name) in state]
        return run_names

    def get_run_id(self, run_name: str) -> str:
        return '-'.join(run_name.split('-')[2:])

    def get_state(self, run_name: str) -> str:
        try:
            return self.load_metadata(run_name)['state']
        except FileNotFoundError:
            if (self._root / run_name).exists():
                return 'error'
            else:
                raise

    def load_metadata(self, run_name: str) -> dict:
        with (self._root / run_name / 'files' / 'wandb-metadata.json').open('r') as f:
            items = json.load(f)
        return items


if __name__ == "__main__":
    pass
