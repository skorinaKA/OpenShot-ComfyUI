import argparse
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np


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


def load_pcm16_wav(path):
    with wave.open(path, "rb") as wav_file:
        channels = int(wav_file.getnchannels())
        sample_width = int(wav_file.getsampwidth())
        sample_rate = int(wav_file.getframerate())
        frame_count = int(wav_file.getnframes())
        if sample_width != 2:
            raise RuntimeError("Expected 16-bit PCM WAV, got sample width {}".format(sample_width))
        raw = wav_file.readframes(frame_count)
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    audio = audio.reshape(-1, channels).T
    audio /= 32768.0
    return audio, sample_rate


def save_pcm16_wav(path, audio, sample_rate):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = np.round(audio * 32767.0).astype("<i2").T
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(int(audio.shape[0]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm.tobytes())


def patch_torchaudio_load():
    import torch
    import torchaudio

    def _safe_load(path, *args, **kwargs):
        audio_np, sample_rate = load_pcm16_wav(path)
        waveform = torch.from_numpy(audio_np)
        return waveform, sample_rate

    torchaudio.load = _safe_load


def patch_strip_silence():
    import audiosr.utils as audiosr_utils

    def _no_strip(input_path, temp_path, save_path):
        del input_path
        os.replace(temp_path, save_path)

    audiosr_utils.strip_silence = _no_strip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", required=True, choices=["basic", "speech"])
    parser.add_argument("--release-model", action="store_true")
    args = parser.parse_args()

    import torch

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="openshot_audiosr_")
    try:
        source_wav = os.path.join(tmp_dir, "source.wav")
        enhanced_wav = os.path.join(tmp_dir, "enhanced.wav")
        decode_audio_to_wav(input_path, source_wav)

        patch_torchaudio_load()
        from audiosr import build_model, super_resolution

        patch_strip_silence()

        device = "auto"
        model = build_model(model_name=args.model_name, device=device)
        waveform = super_resolution(
            model,
            source_wav,
            seed=42,
            guidance_scale=3.5,
            ddim_steps=50,
            latent_t_per_second=12.8,
        )

        out_np = np.asarray(waveform, dtype=np.float32)
        if out_np.ndim == 3:
            out_np = out_np[0]
        if out_np.ndim == 1:
            out_np = out_np.reshape(1, -1)
        save_pcm16_wav(enhanced_wav, out_np, 48000)
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
