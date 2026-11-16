"""Every module in the package must import on each supported Python version."""

import importlib
import pathlib

import pytest

import world_model_lens

PACKAGE_ROOT = pathlib.Path(world_model_lens.__file__).parent
MODULES = sorted(
    ".".join(("world_model_lens",) + path.relative_to(PACKAGE_ROOT).with_suffix("").parts).replace(
        ".__init__", ""
    )
    for path in PACKAGE_ROOT.rglob("*.py")
)

OPTIONAL_DEPENDENCIES = {
    "world_model_lens.collaboration": "zarr",
    "world_model_lens.deployment.api": "fastapi",
    "world_model_lens.deployment.demo": "gradio",
}

KNOWN_BROKEN = {
    "world_model_lens.backends.diffusion": "uses a WorldModelMetadata API that does not exist",
}


def _param(module):
    if module in KNOWN_BROKEN:
        mark = pytest.mark.xfail(reason=KNOWN_BROKEN[module], raises=ImportError, strict=True)
        return pytest.param(module, marks=mark)
    return module


@pytest.mark.parametrize("module", [_param(m) for m in MODULES])
def test_module_imports(module):
    for prefix, dependency in OPTIONAL_DEPENDENCIES.items():
        if module == prefix or module.startswith(prefix + "."):
            pytest.importorskip(dependency)
    importlib.import_module(module)
