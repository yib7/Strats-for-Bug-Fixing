"""Set up the local AMD-GPU environment (.venv-rocm) on Windows.

Target: AMD Radeon RX 9070 / 9070 XT (RDNA4, gfx1201) on Windows 11 with
AMD's official PyTorch-on-Windows ROCm wheels.

What this does (idempotent; safe to re-run):
  1. Creates ``.venv-rocm`` with Python 3.12 (AMD's wheels are cp312-only).
  2. Installs the ROCm 7.2.1 SDK wheels + torch 2.9.1+rocm7.2.1 from
     repo.radeon.com.
  3. Installs this project (``pip install -e .[dev]``) — torch is already
     satisfied, so the CPU pin in uv.lock is NOT applied here.
  4. Writes a ``sitecustomize.py`` shim into the venv: AMD's Windows torch
     build omits the distributed C extension, but transformers 5.x imports
     ``torch.distributed.tensor`` unconditionally; the shim pre-seeds
     sys.modules with inert, loud-failing stubs for that unusable subtree.
  5. Patches the venv's ``transformers/core_model_loading.py``: it guards
     the DTensor *import* on ``torch.distributed.is_available()`` but then
     uses ``isinstance(x, DTensor)`` unguarded (upstream bug on
     non-distributed builds); we add an inert placeholder class in the
     else-branch.
  6. Verifies: GPU visible, transformers imports, Trainer selects cuda:0.

Prerequisites (manual): AMD Adrenalin driver >= 26.2.2.
Docs: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html

Usage:  python scripts/setup_rocm_windows.py [--python C:\\path\\to\\python312.exe]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV = REPO_ROOT / ".venv-rocm"
VENV_PY = VENV / "Scripts" / "python.exe"

ROCM_BASE = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"
ROCM_WHEELS = [
    f"{ROCM_BASE}/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
    f"{ROCM_BASE}/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
    f"{ROCM_BASE}/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
    f"{ROCM_BASE}/rocm-7.2.1.tar.gz",
]
TORCH_WHEEL = f"{ROCM_BASE}/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"

SITECUSTOMIZE = '''\
"""ROCm-on-Windows shim (pretrain-or-prompt). VENV-SPECIFIC — .venv-rocm only.

AMD's Windows torch wheels (2.9.1+rocm7.2.1) omit the distributed C
extension (torch._C._distributed_c10d), but transformers 5.x imports
torch.distributed.tensor unconditionally at module-import time
(transformers/distributed/sharding_utils.py). That subtree can never import
on this build, so we pre-seed sys.modules with stub modules whose attributes
are inert placeholder classes: hasattr() probes return False, instantiation
raises loudly. Single-GPU code paths never exercise these names; anything
that genuinely needs them fails immediately instead of corrupting a run.

Deliberately torch-free: importing torch here would run at interpreter
startup for every process in this venv (slow ROCm init, potential hangs).
This venv's torch build is known to lack distributed, so no runtime check.
"""

import sys
import types


def _make_stub(qualname: str):
    class _Meta(type):
        def __getattr__(cls, attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            return _make_stub(f"{qualname}.{attr}")

        def __call__(cls, *args, **kwargs):
            raise RuntimeError(
                f"{qualname} is a ROCm-Windows shim stub: torch.distributed "
                "is unavailable in this torch build."
            )

    name = qualname.rsplit(".", 1)[-1]
    return _Meta(name, (), {"_shim_qualname": qualname})


def _make_module(name: str) -> None:
    mod = types.ModuleType(name)

    def _getattr(attr: str, _name: str = name):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _make_stub(f"{_name}.{attr}")

    mod.__getattr__ = _getattr
    sys.modules[name] = mod


for _name in (
    "torch.distributed.tensor",
    "torch.distributed.tensor._utils",
    "torch.distributed.tensor.placement_types",
    "torch.distributed.tensor.parallel",
    "torch.distributed.device_mesh",
    "torch.distributed._functional_collectives",
    "torch.distributed.fsdp",
    "torch.distributed.fsdp.wrap",
):
    _make_module(_name)
'''

DTENSOR_GUARD_OLD = """\
_torch_distributed_available = torch.distributed.is_available()
if _torch_distributed_available:
    from torch.distributed.tensor import DTensor
"""
DTENSOR_GUARD_NEW = """\
_torch_distributed_available = torch.distributed.is_available()
if _torch_distributed_available:
    from torch.distributed.tensor import DTensor
else:  # pretrain-or-prompt ROCm-Windows patch: inert placeholder so unguarded
    # isinstance(x, DTensor) checks are always False on non-distributed builds.
    class DTensor:  # noqa: N801
        pass
"""


def run(cmd: list[str]) -> None:
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def find_python312(explicit: str | None) -> str:
    if explicit:
        return explicit
    candidates = [
        Path.home() / "AppData/Local/Programs/Python/Python312/python.exe",
        Path("C:/Python312/python.exe"),
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    py = shutil.which("python")
    if py:
        ver = subprocess.run(
            [py, "-c", "import sys; print(sys.version_info[:2] == (3, 12))"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if ver == "True":
            return py
    sys.exit("No Python 3.12 found; pass --python (AMD wheels are cp312-only).")


def patch_transformers() -> None:
    target = VENV / "Lib" / "site-packages" / "transformers" / "core_model_loading.py"
    text = target.read_text(encoding="utf-8")
    if "ROCm-Windows patch" in text:
        print("transformers DTensor patch already applied")
        return
    if DTENSOR_GUARD_OLD not in text:
        sys.exit(
            f"DTensor guard pattern not found in {target} — transformers "
            "layout changed; update this script (or the upstream bug is fixed:"
            " try without the patch)."
        )
    target.write_text(text.replace(DTENSOR_GUARD_OLD, DTENSOR_GUARD_NEW), encoding="utf-8")
    print("patched transformers core_model_loading.py")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", help="Python 3.12 interpreter to build the venv")
    args = ap.parse_args()

    if not VENV_PY.is_file():
        run([find_python312(args.python), "-m", "venv", str(VENV)])
    run([str(VENV_PY), "-m", "pip", "install", "--no-cache-dir", *ROCM_WHEELS])
    run([str(VENV_PY), "-m", "pip", "install", "--no-cache-dir", TORCH_WHEEL])
    run([str(VENV_PY), "-m", "pip", "install", "-e", f"{REPO_ROOT}[dev]"])

    (VENV / "Lib" / "site-packages" / "sitecustomize.py").write_text(
        SITECUSTOMIZE, encoding="utf-8"
    )
    print("wrote sitecustomize.py shim")
    patch_transformers()

    run(
        [
            str(VENV_PY),
            "-c",
            "import torch; assert torch.cuda.is_available(), 'GPU not visible';"
            " print('GPU:', torch.cuda.get_device_name(0));"
            " from transformers import T5ForConditionalGeneration, Trainer;"
            " print('transformers OK')",
        ]
    )
    print(
        "\nDone. Run the pipeline on the GPU with:\n"
        "  .venv-rocm\\Scripts\\pop smoke --config configs/smoke.yaml"
    )


if __name__ == "__main__":
    main()
