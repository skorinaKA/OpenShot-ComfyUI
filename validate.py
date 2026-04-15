import importlib
import os
import shutil
import subprocess
import sys
try:
    from .audiosr_bootstrap import audiosr_runner_path, ensure_audiosr_environment
except Exception:
    from audiosr_bootstrap import audiosr_runner_path, ensure_audiosr_environment


BASE_REQUIRED_MODULES = [
    ("numpy", "numpy"),
    ("PIL", "Pillow"),
    ("hydra", "hydra-core"),
    ("decord", "decord"),
    ("transformers", "transformers"),
    ("timm", "timm"),
    ("transnetv2_pytorch", "transnetv2-pytorch"),
    ("torchaudio", "torchaudio"),
    ("sam2", "sam2"),
]

COMFY_REQUIRED_MODULES = [
    ("torch", "PyTorch / ComfyUI runtime"),
    ("comfy.model_management", "ComfyUI runtime"),
    ("folder_paths", "ComfyUI runtime"),
]

EXPECTED_NODES = [
    "OpenShotTransNetSceneDetect",
    "OpenShotDownloadAndLoadSAM2Model",
    "OpenShotSam2Segmentation",
    "OpenShotSam2VideoSegmentationAddPoints",
    "OpenShotSam2VideoSegmentationChunked",
    "OpenShotImageBlurMasked",
    "OpenShotImageHighlightMasked",
    "OpenShotDeepFilterNetDenoiseAudio",
    "OpenShotAudioSRClarity",
    "OpenShotGroundingDinoDetect",
    "OpenShotSceneRangesFromSegments",
]


def check_module(import_name, label):
    try:
        importlib.import_module(import_name)
        print("[OK]   import {} ({})".format(import_name, label))
        return True
    except Exception as ex:
        print("[FAIL] import {} ({}): {}".format(import_name, label, ex))
        return False


def check_binary(name):
    path = shutil.which(name)
    if path:
        print("[OK]   binary {} -> {}".format(name, path))
        return True
    print("[FAIL] binary {} not found on PATH".format(name))
    return False


def check_nodes_module():
    try:
        nodes = importlib.import_module("nodes")
        mapping = getattr(nodes, "NODE_CLASS_MAPPINGS", {})
        missing = [name for name in EXPECTED_NODES if name not in mapping]
        if missing:
            print("[FAIL] nodes.py imported, but missing node mappings: {}".format(", ".join(missing)))
            return False
        print("[OK]   nodes.py imported with {} registered nodes".format(len(mapping)))
        return True
    except Exception as ex:
        print("[FAIL] nodes.py import failed: {}".format(ex))
        return False


def module_available(import_name):
    try:
        importlib.import_module(import_name)
        return True
    except Exception:
        return False


def check_deepfilternet_runner():
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepfilternet_runner.py")
    if not os.path.isfile(runner):
        print("[FAIL] deepfilternet runner missing: {}".format(runner))
        return False
    code = (
        "import importlib.util, sys;"
        "spec=importlib.util.spec_from_file_location('openshot_df_runner', sys.argv[1]);"
        "mod=importlib.util.module_from_spec(spec);"
        "spec.loader.exec_module(mod);"
        "mod.ensure_torchaudio_backend_compat();"
        "import df.enhance;"
        "print('ok')"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code, runner],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("[OK]   deepfilternet runner compatibility")
        return True
    except subprocess.CalledProcessError as ex:
        err = (ex.stderr or ex.stdout or "").strip()
        print("[FAIL] deepfilternet runner compatibility: {}".format(err or "unknown error"))
        return False


def check_audiosr_runner():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    runner = audiosr_runner_path(base_dir)
    if not os.path.isfile(runner):
        print("[FAIL] audiosr runner missing: {}".format(runner))
        return False
    if not module_available("torch") or not module_available("torchaudio") or not module_available("torchvision"):
        print("[OK]   audiosr runner present (skipping isolated env probe; main torch/torchaudio/torchvision not available)")
        return True
    try:
        python_path = ensure_audiosr_environment(base_dir)
    except Exception as ex:
        print("[FAIL] audiosr isolated env bootstrap: {}".format(ex))
        return False

    code = (
        "import warnings; warnings.filterwarnings('ignore');"
        "from audiosr import build_model, super_resolution;"
        "print('ok')"
    )
    try:
        subprocess.run(
            [python_path, "-c", code],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("[OK]   audiosr isolated env import compatibility")
        return True
    except subprocess.CalledProcessError as ex:
        err = "\n".join(part.strip() for part in ((ex.stdout or ""), (ex.stderr or "")) if part.strip())
        print("[FAIL] audiosr isolated env import compatibility: {}".format(err or "unknown error"))
        return False


def main():
    ok = True

    print("OpenShot-ComfyUI validation")
    print("===========================")

    for import_name, label in BASE_REQUIRED_MODULES:
        ok = check_module(import_name, label) and ok

    ok = check_deepfilternet_runner() and ok
    ok = check_audiosr_runner() and ok

    for binary in ("ffmpeg", "ffprobe"):
        ok = check_binary(binary) and ok

    comfy_available = module_available("torch") and module_available("comfy.model_management") and module_available("folder_paths")
    if comfy_available:
        print("\nComfyUI runtime detected.")
        for import_name, label in COMFY_REQUIRED_MODULES:
            ok = check_module(import_name, label) and ok
        ok = check_nodes_module() and ok
    else:
        print("\nComfyUI runtime not detected; skipping Comfy-specific checks.")
        print("Run this script inside the ComfyUI Python environment for full node registration validation.")

    if ok:
        print("\nValidation passed.")
        return 0

    print("\nValidation failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
