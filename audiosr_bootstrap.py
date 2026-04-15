import os
import subprocess
import time
import venv


AUDIOSR_ENV_VERSION = "6"


def audiosr_env_dir(base_dir):
    path = os.path.join(base_dir, ".openshot_envs", "audiosr")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def audiosr_python_path(base_dir):
    env_dir = audiosr_env_dir(base_dir)
    if os.name == "nt":
        return os.path.join(env_dir, "Scripts", "python.exe")
    return os.path.join(env_dir, "bin", "python")


def audiosr_runner_path(base_dir):
    return os.path.join(base_dir, "audiosr_runner.py")


def run_checked(cmd, error_prefix):
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as ex:
        err = "\n".join(part.strip() for part in ((ex.stdout or ""), (ex.stderr or "")) if part.strip())
        if len(err) > 4000:
            err = err[:2000] + "\n...(truncated)...\n" + err[-1500:]
        raise RuntimeError("{}: {}".format(error_prefix, err))


def audiosr_env_needs_refresh(marker_path, python_path):
    if not os.path.isfile(marker_path) or not os.path.isfile(python_path):
        return True
    try:
        with open(marker_path, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle.readlines() if line.strip()]
    except Exception:
        return True
    return (not lines) or lines[0] != AUDIOSR_ENV_VERSION


def ensure_audiosr_environment(base_dir):
    env_dir = audiosr_env_dir(base_dir)
    python_path = audiosr_python_path(base_dir)
    marker_path = os.path.join(env_dir, ".ready")
    runner_path = audiosr_runner_path(base_dir)

    if not audiosr_env_needs_refresh(marker_path, python_path):
        return python_path

    builder = venv.EnvBuilder(with_pip=True, system_site_packages=True)
    if not os.path.isdir(env_dir):
        builder.create(env_dir)
    elif not os.path.isfile(python_path):
        builder.create(env_dir)

    run_checked([python_path, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], "AudioSR pip bootstrap failed")
    run_checked(
        [
            python_path,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--no-deps",
            "audiosr==0.0.7",
        ],
        "AudioSR core package install failed",
    )
    run_checked(
        [
            python_path,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "numpy<=1.23.5",
            "librosa==0.9.2",
            "transformers==4.30.2",
            "soundfile",
            "phonemizer",
            "torchlibrosa>=0.0.9",
            "tqdm",
            "progressbar",
            "ipdb",
            "dlinfo",
            "segments",
            "csvw",
            "language-tags",
            "ftfy",
            "einops",
            "pandas",
            "unidecode",
            "chardet",
            "pyyaml",
            "gradio",
            "huggingface-hub",
            "scipy",
            "timm",
        ],
        "AudioSR dependency install failed",
    )

    with open(marker_path, "w", encoding="utf-8") as handle:
        handle.write("{}\n".format(AUDIOSR_ENV_VERSION))
        handle.write("{}\n".format(time.time()))
    if not os.path.isfile(runner_path):
        raise RuntimeError("AudioSR runner script not found: {}".format(runner_path))
    return python_path

