# OpenShot-ComfyUI

OpenShot-ComfyUI provides production-focused ComfyUI nodes built for OpenShot integration, with a strong focus on reliable SAM2 workflows for longer videos.

The goal is simple: make advanced segmentation and video analysis features feel native inside OpenShot's UI, while keeping the underlying Comfy graphs stable, predictable, and memory-safe.

## Why this exists

OpenShot needs SAM2 pipelines that can handle real-world clips, not just short demos.

Many SAM2 custom-node workflows process or retain full-video state in ways that become fragile or memory-heavy as clip length grows. In practice, that can lead to slowdowns, failures, or OOM behavior on longer timelines.

This project addresses that gap with chunk-oriented processing designed specifically for OpenShot's planned UI integration path.

## How this works

- Keep node interfaces close to standard ComfyUI types and patterns.
- Process video segmentation in bounded chunks instead of retaining full-video mask history.
- Return outputs that are easier for OpenShot to consume and orchestrate in larger editing workflows.
- Include practical companion nodes (GroundingDINO + TransNetV2) that support automated, timeline-aware tooling.

## What this includes (V1)

- `OpenShotDownloadAndLoadSAM2Model`
- `OpenShotSam2Segmentation` (single-image)
- `OpenShotSam2VideoSegmentationAddPoints`
- `OpenShotSam2VideoSegmentationChunked` (meta-batch/chunk friendly)
- `OpenShotGroundingDinoDetect` (text-prompted object detection -> mask + JSON)
- `OpenShotTransNetSceneDetect` (direct TransNetV2 inference -> IN/OUT JSON ranges)
- `OpenShotDeepFilterNetDenoiseAudio` (file-path based audio denoise -> FLAC path)
- `OpenShotLavaSRSpeechClarity` (LavaSR speech runner -> FLAC path)

## Attribution

This project is inspired by and partially based on ideas and APIs from:

- `kijai/ComfyUI-segment-anything-2`
- Meta SAM2 research/code

Please see upstream projects for full original implementations and credits.

## Requirements

- ComfyUI
- PyTorch (as used by your Comfy install)
- `ffmpeg` and `ffprobe` available on your `PATH`
- `git` available on your `PATH` for installing LavaSR from `requirements.txt`

Install this node pack into `ComfyUI/custom_nodes/OpenShot-ComfyUI` and restart ComfyUI.

## Quick install (copy/paste)

Install this node pack's Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Install SAM2 separately:

```bash
python -m pip install --no-build-isolation git+https://github.com/facebookresearch/sam2.git
```

Validate the environment:

```bash
python validate.py
```

Restart ComfyUI after install.

`validate.py` supports two cases:

- On a regular laptop/dev environment, it validates the Python packages plus `ffmpeg`/`ffprobe`
- Inside the actual ComfyUI Python environment, it also validates Comfy imports and node registration

If you run it in the ComfyUI environment and it passes, restart ComfyUI and the nodes should load.

SAM2 is installed separately on purpose. Keeping it out of `requirements.txt` makes the normal dependency install much more reliable and avoids long hangs during pip's build-isolation step.

## First-use model downloads

- `OpenShotDownloadAndLoadSAM2Model` downloads supported SAM2 checkpoints into `ComfyUI/models/sam2` on first use.
- `OpenShotGroundingDinoDetect` downloads model weights from Hugging Face on first use and uses the normal HF cache.
- `OpenShotDeepFilterNetDenoiseAudio` downloads the default `DeepFilterNet3` model on first use using DeepFilterNet's cache directory.
- `OpenShotLavaSRSpeechClarity` downloads the `YatharthS/LavaSR` model snapshot from Hugging Face on first run.
- `OpenShotTransNetSceneDetect` does not require a separate manual weight download from this node pack.

## Audio denoise node

`OpenShotDeepFilterNetDenoiseAudio` is intentionally minimal:

- Input: `source_audio_path`, `noise_reduction`, `keep_model_loaded`
- Output: a new FLAC file path
- The node uses the upstream DeepFilterNet package and blends the enhanced result with the original signal using `noise_reduction` from `0.0` to `1.0`
- `0.0` means "keep the original audio" and still emits a FLAC copy
- `1.0` means "full denoise"

The node accepts typical audio formats that `ffmpeg` can decode and always writes a new `.flac` file into ComfyUI's output folder under `openshot_audio/`.

## Speech clarity node

`OpenShotLavaSRSpeechClarity` is intended for low-quality speech recordings:

- Input: `source_audio_path` or `audio`, `keep_model_loaded`
- Output: a new FLAC file path
- Uses LavaSR v2 speech enhancement
- Intended for speech/dialogue clarity, not general music restoration

LavaSR is installed through `requirements.txt` in the main Comfy environment.
The first `Clarity -> Speech` run will still take longer because LavaSR downloads its model snapshot from Hugging Face on demand.

## Validation script

Run:

```bash
python validate.py
```

It checks:

- required Python imports
- DeepFilterNet compatibility through the bundled runner shim
- `ffmpeg` and `ffprobe`
- ComfyUI-side imports needed for node registration, when ComfyUI is available
- that the expected node classes are present, when ComfyUI is available

## Notes

- `OpenShotSam2VideoSegmentationChunked` returns only the requested chunk range (bounded memory) instead of collecting whole-video masks.
- For very long videos, pair chunked outputs with batch-safe downstream nodes (VHS meta-batch, staged processing, or on-disk intermediates).
- `torchaudio` is listed explicitly because DeepFilterNet imports it internally and some environments do not pull it in automatically.
- LavaSR preserves the original channel count by processing each channel independently before recombining the output.

---

Copyright (C) 2026 OpenShot Studios, LLC

Licensed under the GNU General Public License v3.0 (GPLv3).
See [LICENSE.md](LICENSE.md) for the full license text.
