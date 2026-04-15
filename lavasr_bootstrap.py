import os
import subprocess
import time
import venv


LAVASR_ENV_VERSION = "1"


def _log(message):
    print("[OpenShot-ComfyUI:LavaSR] {}".format(message), flush=True)


def lavasr_env_dir(base_dir):
    path = os.path.join(base_dir, ".openshot_envs", "lavasr")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def lavasr_python_path(base_dir):
    env_dir = lavasr_env_dir(base_dir)
    if os.name == "nt":
        return os.path.join(env_dir, "Scripts", "python.exe")
    return os.path.join(env_dir, "bin", "python")


def lavasr_runner_path(base_dir):
    return os.path.join(base_dir, "lavasr_runner.py")


def run_checked(cmd, error_prefix):
    try:
        _log("Running: {}".format(" ".join(str(part) for part in cmd)))
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines = []
        assert proc.stdout is not None
        for line in proc.stdout:
            text = line.rstrip()
            if text:
                print(text, flush=True)
                lines.append(text)
        returncode = proc.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd, output="\n".join(lines), stderr="")
    except subprocess.CalledProcessError as ex:
        err = "\n".join(part.strip() for part in ((ex.output or ""), (ex.stderr or "")) if part.strip())
        if len(err) > 4000:
            err = err[:2000] + "\n...(truncated)...\n" + err[-1500:]
        raise RuntimeError("{}: {}".format(error_prefix, err))


def lavasr_env_needs_refresh(marker_path, python_path):
    if not os.path.isfile(marker_path) or not os.path.isfile(python_path):
        return True
    try:
        with open(marker_path, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle.readlines() if line.strip()]
    except Exception:
        return True
    return (not lines) or lines[0] != LAVASR_ENV_VERSION


def ensure_lavasr_environment(base_dir):
    env_dir = lavasr_env_dir(base_dir)
    python_path = lavasr_python_path(base_dir)
    marker_path = os.path.join(env_dir, ".ready")
    runner_path = lavasr_runner_path(base_dir)

    if not lavasr_env_needs_refresh(marker_path, python_path):
        _log("Using existing isolated environment: {}".format(env_dir))
        return python_path

    _log("Preparing isolated environment: {}".format(env_dir))
    builder = venv.EnvBuilder(with_pip=True, system_site_packages=True)
    if not os.path.isdir(env_dir):
        _log("Creating virtual environment")
        builder.create(env_dir)
    elif not os.path.isfile(python_path):
        _log("Recreating missing Python inside virtual environment")
        builder.create(env_dir)

    _log("Bootstrapping pip/setuptools/wheel")
    run_checked([python_path, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], "LavaSR pip bootstrap failed")
    _log("Installing LavaSR")
    run_checked(
        [
            python_path,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "git+https://github.com/ysharma3501/LavaSR.git",
            "huggingface-hub",
        ],
        "LavaSR dependency install failed",
    )

    with open(marker_path, "w", encoding="utf-8") as handle:
        handle.write("{}\n".format(LAVASR_ENV_VERSION))
        handle.write("{}\n".format(time.time()))
    if not os.path.isfile(runner_path):
        raise RuntimeError("LavaSR runner script not found: {}".format(runner_path))
    _log("Isolated environment ready")
    return python_path

