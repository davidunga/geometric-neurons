from pathlib import Path

_project_root = Path(__file__).parent
_project_name = _project_root.name
_resources_root = _project_root / "resources"
_output_root = _project_root / "outputs"

CV_DIR = _output_root / "cv"
MODELS_DIR = _output_root / "models"
DATA_DIR = _resources_root / "data"
CONFIG_DIR = _resources_root / "config"
TENSORBOARD_DIR = Path("~/tensorboard").expanduser() / _project_name
GLOBAL_DATA_DIR = Path("~/data").expanduser()
