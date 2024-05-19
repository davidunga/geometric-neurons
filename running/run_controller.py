from paths import RUNNING_DIR
import yaml


def safe_load_settings() -> dict:
    try:
        return yaml.safe_load((RUNNING_DIR / 'run_settings.yml').open('r'))
    except:
        return {}


