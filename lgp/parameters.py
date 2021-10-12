import os
import json
import lgp.paths

EXPERIMENT_DEFINITION = None


def assert_present(hparams, required_params):
    for h in required_params:
        assert hasattr(hparams, h), f"Required hyperparameters: {required_params}. {h} is missing!"


def dict_merge(dict1, dict2):
    outdict = dict1.copy()
    for k,v in dict2.items():
        if k in dict1 and isinstance(dict1[k], dict) and isinstance(v, dict):
            outdict[k] = dict_merge(dict1[k], v)
        else:
            outdict[k] = dict2[k]
    return outdict


def resolve_includes(definition):
    includes_only = {}
    if "@include" in definition:
        # Load each of the included sets of parameters, overlaying over the the ones before
        for incl in definition["@include"]:
            overlay = load_experiment_definition(incl)
            includes_only = dict_merge(includes_only, overlay)
        del definition["@include"]
    # Overlay the rest of the definition over the included parameters
    definition = dict_merge(includes_only, definition)
    return definition


def load_experiment_definition(name):
    json_path = os.path.join(os.path.dirname(__file__), "experiment_definitions", f"{name}.json")
    if not os.path.isfile(json_path):
        raise ValueError(f"The experiment definition is not found at {json_path}")
    with open(json_path, "r") as fp:
        definition = json.load(fp)
    definition = resolve_includes(definition)

    global EXPERIMENT_DEFINITION
    EXPERIMENT_DEFINITION = definition
    return definition


def get_experiment_definition():
    global EXPERIMENT_DEFINITION
    assert EXPERIMENT_DEFINITION is not None, "Experiment definiton not loaded. Please call load_experiment_definiton first"
    return Hyperparams(EXPERIMENT_DEFINITION)


class Hyperparams:
    def __init__(self, d):
        self.d = d
        for k, v in self.d.items():
            if isinstance(v, dict):
                self.d[k] = Hyperparams(v)

    def get(self, attribute, default=None):
        if attribute in self.d:
            return self.d[attribute]
        else:
            return default

    def __getattr__(self, item):
        if item == "d":
            # This is necessary to to not get stuck in recursion before __init__ is called.
            raise AttributeError()
        elif item in self.d:
            return self.d[item]
        else:
            raise AttributeError(f"Hyperparameters don't contain: {item}")