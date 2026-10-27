# Video Batch Inference on Serverless GPU (Ray + vLLM)

Serverless port of [`3_Video_Batch_Inference_Ray_Core`](../3_Video_Batch_Inference_Ray_Core). It runs a vision-language model (Qwen2.5-VL) over a folder of CCTV videos with Ray Data and vLLM, turns the model's descriptions into a structured Delta table with AI Functions, and produces a face-redacted frame for presentations.

No classic GPU cluster, init scripts, or `apt-get` are needed. Only notebook 02 uses a GPU (one H100 on serverless GPU compute).

## Notebooks

| Notebook | Compute | What it does | Output | Typical time |
| --- | --- | --- | --- | --- |
| `00 - Download Video Data` | Serverless CPU | Downloads the Kaggle CCTV dataset (one category by default) into a UC Volume and records each video's duration with PyAV | `videos_file_reference` table | ~10 min per category |
| `01 - Download and Register Models` | Serverless CPU | Downloads the Qwen2.5-VL weights and registers them in Unity Catalog with the `production` alias | UC model `qwen2_5_vl-7b` | ~30 min (7B) |
| `02 - Batch Video Inference on Ray + vLLM` | **Serverless GPU, H100**, base environment **Databricks AI** (v6) | Samples frames at ~1 fps (at most 64), runs them through vLLM in batches | `video_inferences` table | ~4 min staging + ~3 min for 49 videos |
| `03 - Structured Entity Extraction` | Serverless CPU or SQL warehouse | `ai_query` with a strict JSON schema turns each description into incident fields (assessment, confidence, location, time range, suspects) | `video_events` table | a few minutes |
| `04 - Redacted Frame Extraction` | Serverless CPU | Extracts one frame, pixelates detected faces (OpenCV YuNet + Haar), saves a PNG | PNG in the Volume | < 1 min |

Run them in order. Every notebook has `CATALOG` / `SCHEMA` / `VOLUME` widgets (default `main.vlm_video`, volume `transcribe-video`); use the same values in all five. Notebook 00 creates the schema and volume if they don't exist.

## Prerequisites

* **Kaggle API token** in a secret scope `kaggle` with keys `username` and `key` (or `~/.kaggle/kaggle.json`).
* **Serverless GPU** with H100 available in the workspace.
* **Foundation Model API** endpoint `databricks-meta-llama-3-3-70b-instruct` (set `LLM_ENDPOINT` in 03 to use another).
* Permission to create a schema, volume, tables, and a registered model in the target catalog.

## Setting up notebook 02 compute

1. **Connect** -> **Serverless GPU**.
2. In the **Environment** panel, set **Accelerator** to **H100** and the base environment to **Databricks AI** (v6), then apply.
3. Run the install cell. It installs only `av`; torch, vLLM, Ray, and transformers come with the environment.

Set `MAX_VIDEOS` to 3 for a quick smoke test.

## Results (49 shoplifting videos, Qwen2.5-VL-7B, 1xH100)

* 49 of 49 videos processed, 0 decode failures, ~3 min of inference.
* Assessments from notebook 03: 22 NORMAL, 12 LIKELY, 10 POSSIBLE, 3 INSUFFICIENT_EVIDENCE, 2 CONFIRMED. Every video in this set is labelled shoplifting, so 24 of 49 (49%) were flagged as CONFIRMED, LIKELY, or POSSIBLE.
* The 7B model is more cautious than the 32B model used in the classic version. For a larger model, see below.

## Using the 32B model

1. In 01, set `HF_MODEL_ID` = `Qwen/Qwen2.5-VL-32B-Instruct`, `MODEL_NAME` = `qwen2_5_vl-32b`, and `STAGING_DIR` to a Volume path (for example `/Volumes/main/vlm_video/transcribe-video/hf_staging`). The weights are ~66 GB; logging makes a second copy, and both don't fit on local disk.
2. In 02, set `MODEL_NAME` = `qwen2_5_vl-32b` and one of:
   * `QUANTIZATION` = `fp8` on 1xH100 (weights quantized to ~33 GB at load time), or
   * `TENSOR_PARALLEL` = `2` on a multi-GPU node (bf16, sharded over two GPUs per vLLM engine).

The 32B settings have not yet been validated end to end on serverless.

## Differences from the classic version

* **Correct timestamps.** Each prompt gives vLLM the real sampling rate (`video_metadata`), and the prompt states the video length and frame interval. The classic notebook always passed `fps=1`.
* **Real batching.** Each `generate()` call processes a batch of videos.
* **No system ffmpeg.** PyAV replaces `ffmpeg` / `ffprobe`.
* **Model registration without loading it.** 01 downloads the weight files and registers them as a pyfunc artifact. The classic version loaded the model into memory on a 672 GB driver.
* **Structured output fix.** 03 lists every field in `required` and sets `strict` on `json_schema`. Before, `location`, `time_range`, and `confidence` were empty on every row. (The classic notebook now has the same fix.)
* **No MediaPipe.** MediaPipe pins `numpy<2` and an old protobuf, which crashes the serverless kernel. 04 uses OpenCV YuNet and Haar cascades on a 3x upscaled frame instead.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `undefined symbol` from flash-attn, or vLLM/torch version mismatches | Something `%pip install`ed vLLM, torch, or Ray over the AI environment. **New Session does not undo it.** Uninstall those packages, or reset the environment in the Environment panel. |
| `FileNotFoundError: 'ninja'` or a FlashInfer build error at engine start | The environment's `bin/` is not on `PATH`. The pipeline cell passes `PATH` and `VLLM_USE_FLASHINFER_SAMPLER=0` to the Ray workers; keep those lines. |
| Ray fails at init with a `/local_disk0/tmp` error | `/local_disk0` is not writable on serverless. The pipeline cell sets `RAY_TMPDIR=/tmp/ray`. |
| `No videos matched ...` in 02 | `CATALOG` / `SCHEMA` differ from the values used in 00, or `CATEGORY_FILTER` doesn't match the downloaded category. |
| `location` / `confidence` empty in `video_events` | An older copy of 03 without the schema fix. Re-run the structuring cell. |
| A face is visible in the redacted frame | Detection is automatic and can miss small faces in low-resolution footage. Check the preview, and pick another frame (`FRAME_AT_SECONDS`) if needed. |

## Known limitations

* A few long videos still run up to `max_tokens` (1024) and are cut off (`finish_reason = length`).
* `time_range` formats vary (`60 - 75 seconds`, `0:00 - 0:23`), because they come from the model's prose.
* Ray prints deprecation warnings for `concurrency=` in `map` / `map_batches`; they are harmless.
