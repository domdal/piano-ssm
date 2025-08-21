from pathlib import Path
import importlib.util
import sys
import types

base_path = Path("S-Edge_PianoSSM/src/model").resolve()
pkg_name = "s_edge_pkg"  # fake package name

# Register the package so relative imports work
pkg = types.ModuleType(pkg_name)
pkg.__path__ = [str(base_path)]
sys.modules[pkg_name] = pkg

def _load_as_submodule(subname, filename):
    full_name = f"{pkg_name}.{subname}"
    spec = importlib.util.spec_from_file_location(full_name, base_path / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod

# Load both modules as part of the fake package
sequence_layer = _load_as_submodule("sequence_layer", "sequence_layer.py")
mimo_ssm = _load_as_submodule("mimo_ssm", "mimo_ssm.py")

# Re-export top-level classes
SequenceLayer = sequence_layer.SequenceLayer
MIMOSSM = mimo_ssm.MIMOSSM
