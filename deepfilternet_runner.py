import argparse
import os
import subprocess
import sys
import tempfile
import types


def ensure_torchaudio_backend_compat():
    import torchaudio

    backend_module = sys.modules.get("torchaudio.backend")
    common_module = sys.modules.get("torchaudio.backend.common")
    if backend_module is not None and common_module is not None:
        return

    audio_meta = None
    try:
        from torchaudio._backend.common import AudioMetaData as _AudioMetaData

        audio_meta = _AudioMetaData
    except Exception:
        audio_meta = getattr(torchaudio, "AudioMetaData", None)
    if audio_meta is None:
        audio_meta = object

    if backend_module is None:
        backend_module = types.ModuleType("torchaudio.backend")
        backend_module.__package__ = "torchaudio.backend"
        backend_module.__path__ = []
        sys.modules["torchaudio.backend"] = backend_module

    if common_module is None:
        common_module = types.ModuleType("torchaudio.backend.common")
        common_module.__package__ = "torchaudio.backend"
        sys.modules["torchaudio.backend.common"] = common_module

    common_module.AudioMetaData = audio_meta
    backend_module.common = common_module
    setattr(torchaudio, "backend", backend_module)


def run_ffmpeg(args, error_prefix):
    cmd = ["ffmpeg", "-y"] + list(args)
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found")
    except subprocess.CalledProcessError as ex:
        err = (ex.stderr or "").strip()
        if len(err) > 1000:
            err = err[:1000] + "...(truncated)"
        raise RuntimeError("{}: {}".format(error_prefix, err))


def decode_audio_to_wav(source_path, wav_path):
    run_ffmpeg(["-i", source_path, "-vn", "-acodec", "pcm_s16le", wav_path], "ffmpeg audio decode failed")


def encode_audio_to_flac(source_path, flac_path):
    run_ffmpeg(["-i", source_path, "-vn", "-c:a", "flac", flac_path], "ffmpeg FLAC encode failed")


def match_audio_length(audio, target_length):
    import torch.nn.functional as F

    target_length = int(max(0, target_length))
    current = int(audio.shape[-1])
    if current == target_length:
        return audio
    if current > target_length:
        return audio[..., :target_length]
    return F.pad(audio, (0, target_length - current))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--amount", type=float, required=True)
    parser.add_argument("--release-model", action="store_true")
    args = parser.parse_args()

    ensure_torchaudio_backend_compat()

    import torch
    import torchaudio
    import torchaudio.functional as ta_functional
    from df.enhance import enhance as df_enhance
    from df.enhance import init_df as df_init_df

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    amount = float(max(0.0, min(1.0, args.amount)))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="openshot_df_runner_")
    try:
        source_wav = os.path.join(tmp_dir, "source.wav")
        enhanced_wav = os.path.join(tmp_dir, "enhanced.wav")
        decode_audio_to_wav(input_path, source_wav)

        source_audio, source_sr = torchaudio.load(source_wav)
        source_audio = source_audio.to(torch.float32)

        if amount <= 0.0:
            torchaudio.save(enhanced_wav, source_audio.cpu(), int(source_sr))
            encode_audio_to_flac(enhanced_wav, output_path)
            return 0

        model, df_state, _suffix = df_init_df(
            model_base_dir=None,
            log_file=None,
            config_allow_defaults=True,
            default_model="DeepFilterNet3",
        )

        model_sr = int(getattr(df_state, "sr", 48000)() if callable(getattr(df_state, "sr", None)) else getattr(df_state, "sr", 48000))
        work_audio = source_audio
        if int(source_sr) != model_sr:
            work_audio = ta_functional.resample(work_audio, int(source_sr), model_sr)

        enhanced_audio = df_enhance(model, df_state, work_audio, pad=True)
        enhanced_audio = (work_audio * (1.0 - amount)) + (enhanced_audio * amount)
        enhanced_audio = torch.clamp(enhanced_audio, -1.0, 1.0)

        if int(source_sr) != model_sr:
            enhanced_audio = ta_functional.resample(enhanced_audio, model_sr, int(source_sr))
        enhanced_audio = match_audio_length(enhanced_audio, int(source_audio.shape[-1]))

        torchaudio.save(enhanced_wav, enhanced_audio.cpu(), int(source_sr))
        encode_audio_to_flac(enhanced_wav, output_path)

        if args.release_model:
            try:
                model.cpu()
            except Exception:
                pass
        return 0
    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
