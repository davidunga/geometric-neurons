from . hatsopoulos import HATSO_DATASET_SPECS as _HATSO_DATASET_SPECS
from copy import deepcopy

_specs = {
    "HATSO": _HATSO_DATASET_SPECS,
}


def get_datasets_specs(flat: bool = True):

    specs = deepcopy(_specs)
    if not flat:
        return specs

    flat_specs = {}
    for dataset_name, dataset_specs in specs.items():
        assert set(dataset_specs.keys()).isdisjoint(flat_specs.keys())
        flat_specs.update(dataset_specs)
    return flat_specs
