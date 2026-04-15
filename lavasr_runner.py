import argparse
import json
import os
import subprocess
import tempfile
import wave

import numpy as np


def log(message):
    print("[OpenShot-ComfyUI:LavaSR] {}".format(message), flush=True)


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


def probe_audio_info(path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,channel_layout",
        "-of",
        "json",
        path,
    ]
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        data = json.loads(result.stdout or "{}")
    except Exception:
        return {}
    streams = data.get("streams") or []
    if not streams:
        return {}
    stream = streams[0] or {}
    return {
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
        "channel_layout": str(stream.get("channel_layout") or "").strip(),
    }


def decode_audio_to_wav(source_path, wav_path):
    run_ffmpeg(["-i", source_path, "-vn", "-acodec", "pcm_s16le", wav_path], "ffmpeg audio decode failed")


def encode_audio_to_flac(source_path, flac_path, channel_layout=None):
    args = ["-i", source_path, "-vn"]
    if str(channel_layout or "").strip():
        args.extend(["-channel_layout", str(channel_layout).strip()])
    args.extend(["-c:a", "flac", flac_path])
    run_ffmpeg(args, "ffmpeg FLAC encode failed")


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
    if channels <= 0:
        raise RuntimeError("Invalid WAV channel count: {}".format(channels))
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


def save_mono_pcm16_wav(path, audio, sample_rate):
    save_pcm16_wav(path, np.asarray(audio, dtype=np.float32).reshape(1, -1), sample_rate)


def is_cuda_oom(ex):
    text = str(ex or "").lower()
    return "outofmemoryerror" in text or "out of memory" in text or "would exceed allowed memory" in text


def safe_input_sr(source_sr):
    source_sr = int(source_sr or 16000)
    return max(8000, min(48000, source_sr))


def run_lavasr_channels(source_audio_np, source_sr, device_name, tmp_dir):
    from LavaSR.model import LavaEnhance2

    input_sr = safe_input_sr(source_sr)
    log("Loading LavaSR speech model on {}".format(device_name))
    lava_model = LavaEnhance2("YatharthS/LavaSR", device_name)
    channel_outputs = []
    for channel_index in range(int(source_audio_np.shape[0])):
        log("Enhancing channel {}/{} on {}".format(channel_index + 1, int(source_audio_np.shape[0]), device_name))
        channel_path = os.path.join(tmp_dir, "channel_{}.wav".format(channel_index))
        save_mono_pcm16_wav(channel_path, source_audio_np[channel_index], int(source_sr))
        channel_audio, _input_sr = lava_model.load_audio(channel_path, input_sr=input_sr)
        output_audio = lava_model.enhance(channel_audio, denoise=False, batch=False).cpu().numpy().reshape(-1)
        channel_outputs.append(output_audio.astype(np.float32))
    max_len = max(int(ch.shape[0]) for ch in channel_outputs)
    padded = []
    for ch in channel_outputs:
        if int(ch.shape[0]) < max_len:
            ch = np.pad(ch, (0, max_len - int(ch.shape[0])), mode="edge")
        padded.append(ch)
    if device_name != "cpu":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return np.stack(padded, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--release-model", action="store_true")
    args = parser.parse_args()

    import torch
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    source_info = probe_audio_info(input_path)

    tmp_dir = tempfile.mkdtemp(prefix="openshot_lavasr_")
    try:
        source_wav = os.path.join(tmp_dir, "source.wav")
        enhanced_wav = os.path.join(tmp_dir, "enhanced.wav")
        log("Decoding input audio with ffmpeg")
        decode_audio_to_wav(input_path, source_wav)
        source_audio_np, source_sr = load_pcm16_wav(source_wav)

        waveform = None
        try:
            preferred_device = "cuda" if torch.cuda.is_available() else "cpu"
            waveform = run_lavasr_channels(source_audio_np, source_sr, preferred_device, tmp_dir)
        except Exception as ex:
            if preferred_device != "cuda" or not is_cuda_oom(ex):
                raise
            log("CUDA OOM during LavaSR; retrying on CPU")
            waveform = run_lavasr_channels(source_audio_np, source_sr, "cpu", tmp_dir)

        log("Encoding enhanced audio to FLAC")
        save_pcm16_wav(enhanced_wav, waveform, 48000)
        encode_audio_to_flac(enhanced_wav, output_path, source_info.get("channel_layout"))
        log("LavaSR output ready: {}".format(output_path))
        return 0
    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
