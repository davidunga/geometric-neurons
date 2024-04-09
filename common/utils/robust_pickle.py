import pickle
import importlib.util


def get_RobustUnpickler(modules):

    class RobustUnpickler(pickle.Unpickler):
        _modules = modules

        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, AttributeError):
                for module_ in self._modules:
                    if hasattr(module_, name):
                        return getattr(module_, name)
                raise ModuleNotFoundError(f"Cannot find class {name} in module {module} or in the specified modules")

    return RobustUnpickler


def get_robust_pickle_module(modules):
    sepc = importlib.util.find_spec('pickle')
    pickle_ = importlib.util.module_from_spec(sepc)
    sepc.loader.exec_module(pickle_)
    pickle_.Unpickler = get_RobustUnpickler(modules)
    return pickle_
