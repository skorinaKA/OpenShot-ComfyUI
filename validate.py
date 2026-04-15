import importlib
import shutil
import sys
import types


BASE_REQUIRED_MODULES = [
    ("numpy", "numpy"),
    ("PIL", "Pillow"),
    ("hydra", "hydra-core"),
    ("decord", "decord"),
    ("transformers", "transformers"),
    ("timm", "timm"),
    ("transnetv2_pytorch", "transnetv2-pytorch"),
    ("df", "deepfilternet"),
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
    "OpenShotGroundingDinoDetect",
    "OpenShotSceneRangesFromSegments",
]


def check_module(import_name, label):
    try:
        if import_name == "df":
            ensure_deepfilternet_torchaudio_compat()
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


def ensure_deepfilternet_torchaudio_compat():
    try:
        torchaudio = importlib.import_module("torchaudio")
    except Exception:
        return
    if "torchaudio.backend.common" in sys.modules:
        return
    compat_audio_meta = None
    try:
        backend_common = importlib.import_module("torchaudio._backend.common")
        compat_audio_meta = getattr(backend_common, "AudioMetaData", None)
    except Exception:
        compat_audio_meta = getattr(torchaudio, "AudioMetaData", None)
    if compat_audio_meta is None:
        return
    module = types.ModuleType("torchaudio.backend.common")
    module.AudioMetaData = compat_audio_meta
    sys.modules["torchaudio.backend.common"] = module


def main():
    ok = True

    print("OpenShot-ComfyUI validation")
    print("===========================")

    for import_name, label in BASE_REQUIRED_MODULES:
        ok = check_module(import_name, label) and ok

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
