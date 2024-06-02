from pathlib import Path

_project_root = Path(__file__).parent
_project_name = _project_root.name
_resources_root = _project_root / "resources"
_output_root = _project_root / "outputs"

ANALYSIS_DIR = _project_root / "analysis"
CV_DIR = _output_root / "cv"
MODELS_DIR = _output_root / "models"
DATA_DIR = _resources_root / "data"
TENSORBOARD_DIR = Path("~/tensorboard").expanduser() / _project_name
GLOBAL_DATA_DIR = Path("~/data").expanduser()
KEYS_DIR = Path('~/keys').expanduser()
WANDB_ROOT = _output_root
RUNNING_DIR = _project_root / "running"
PROCESSED_DIR = _resources_root / "processed"
