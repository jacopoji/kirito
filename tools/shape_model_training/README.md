# Transparent Shape Model — Training Toolkit

Scripts to fine-tune a replacement for `backend/resources/transparent_shape_nms.onnx`,
the detector `TransparentShapeSolver` (`backend/src/solvers/shape.rs`) uses to
solve the "Lie Detector" minigame. There is no training pipeline elsewhere in
this repo — the shipped model is inference-only, baked into the binary by
`backend/build.rs`. This toolkit produces a new `.onnx` that keeps the exact
same input/output contract, so no Rust code needs to change to use it.

This has to run **locally**, outside any sandboxed dev environment: it needs
a Python + CUDA setup and access to your raw recordings. Nothing here trains
automatically — you run the scripts below in order, by hand, with a labeling
pass (CVAT) in between the bootstrap and dataset-prep steps.

## Setup

```
cd tools/shape_model_training
python -m venv .venv && .venv\Scripts\activate   # or source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended for step 4 (training); steps 1-3
are CPU-only and fast.

**Windows note**: `pip install torch` from plain PyPI installs the **CPU-only**
build there (unlike Linux, which bundles CUDA support by default) — `ultralytics`
will happily pull that in as a dependency without complaint, and step 4 will
silently train on CPU instead of your GPU. Check with
`python -c "import torch; print(torch.cuda.is_available())"`; if it prints
`False` on a machine with an NVIDIA GPU, reinstall from PyTorch's CUDA index
(match the CUDA version to `nvidia-smi`'s reported version, e.g. `cu124`):

```
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## The model contract (read this before changing anything)

Reverse-engineered from `backend/src/detect.rs` and replicated in `common.py`
— any exported model MUST match this, or `detect_transparent_shapes` in
`detect.rs` will misread its output:

- **Input**: `[1, 3, 640, 640]` float32, RGB, values in `[0, 1]`, letterboxed
  (uniform-scale resize + 114-gray constant padding — see `preprocess_for_yolo`
  in `detect.rs`).
- **Output**: a single tensor, one row per detection, columns
  `[x1, y1, x2, y2, conf, ...]` in xyxy **letterboxed model-pixel space**
  (extra columns like class id are fine, `detect.rs` only reads the first 5).
  This means the export must be an **NMS-baked** Ultralytics export
  (`model.export(format="onnx", nms=True, ...)`), not raw logits.
- **Single class** (`shape`). The solver's own heuristics in `shape.rs`
  (`track_background_score`, motion-direction divergence from the tracked
  background) decide which detected shape is the actual target — the model
  only needs to find "is there a shape here," not "which one is real."

`04_train_and_export.py` runs an automatic contract self-check after export
so a broken export is caught in Python, before it ever reaches the Rust side.

## Workflow

### 1. Extract candidate frames from your recordings

```
python 01_extract_frames.py \
    --recordings-dir "G:/aaaNiubi/25.0.1/app-debug-mule/dataset/recordings" \
    --resources-dir ../../backend/resources \
    --out-dir ./dataset/raw_frames \
    --sample-every 4
```

Scans each recording with the same popup template-match `detect_lie_detector_shape`
uses (`backend/resources/lie_detector_new_ideal_ratio.png` /
`lie_detector_old_ideal_ratio.png`, threshold 0.6), and crops the fixed
solving region (`title.tl() + (0,20)`, size `755x505` — see
`backend/src/player/solve_shape.rs`) out of every Nth active frame
(`--sample-every`, default 4) to avoid a dataset full of near-duplicates.

Output: `dataset/raw_frames/<recording_name>/frame_NNNNNN.png`, one folder
per recording. Keep the per-recording grouping — step 3 splits train/val by
*session*, not by frame, so near-identical consecutive frames never leak
across the split.

### 2. Bootstrap labels with the current model

```
python 02_bootstrap_labels.py \
    --frames-dir ./dataset/raw_frames \
    --model ../../backend/resources/transparent_shape_nms.onnx \
    --labels-dir ./dataset/raw_labels \
    --conf-threshold 0.15
```

Runs the *current* shipped model over every extracted frame and writes
YOLO-format `.txt` labels. This is the actual accuracy lever: instead of
drawing every box from scratch, you only correct the current model's
mistakes — add boxes it missed, delete false positives, tighten loose boxes.
That directly targets the failure modes that hurt live solving accuracy.

Threshold defaults low (0.15) to bias toward recall — deleting a wrong box
in CVAT is much faster than noticing a missing shape and drawing it fresh.

### 3. Correct the labels in CVAT

Recommended tool: **[CVAT](https://github.com/cvat-ai/cvat)**, self-hosted
locally via Docker — free (MIT-licensed; distinct from the cvat.ai *cloud*
offering, which gates features behind a paid tier) and, thanks to frame
interpolation across near-duplicate consecutive frames, meaningfully faster
to correct in than boxing every frame by hand.

**One-time setup**, from a clone of CVAT *outside* this repo (it's unrelated
tooling, not project source — e.g. a sibling directory to `komari`):

```
git clone --depth 1 https://github.com/cvat-ai/cvat.git
cd cvat
docker compose up -d
```

Then mount this toolkit's `dataset/` directory read-only into CVAT so it can
read frames directly from the share, with no browser upload step at all.
Create `docker-compose.override.yml` next to CVAT's `docker-compose.yml`:

```yaml
services:
  cvat_server: &share-config
    volumes:
      - "<absolute path to>/tools/shape_model_training/dataset:/home/django/share:ro"
  cvat_worker_chunks: *share-config
  cvat_worker_import: *share-config
  cvat_worker_export: *share-config
```

`docker compose up -d` again to apply it, then create your admin login:
`docker compose exec cvat_server python3 ~/manage.py createsuperuser`.

**Creating tasks and pushing bootstrap labels** — `cvat_setup.py` (in this
toolkit, requires `pip install cvat-sdk`) automates the whole thing per
session: creates the CVAT task sourcing images from the mounted share (no
manual upload), then pushes that session's `raw_labels/<session>/` bootstrap
boxes directly via the API (this bypasses CVAT's YOLO zip-import format
entirely, which has a fussier layout than it's worth reverse-engineering for
a single-class use case). Idempotent — safe to re-run; it skips tasks/labels
that already exist.

```
python cvat_setup.py --username admin --password '...' ld_1b03Gfsb   # one session
python cvat_setup.py --username admin --password '...'               # every session
```

**Correcting** — open the task's job in the browser
(`http://localhost:8080`), delete false positives, add missed shapes, nudge
loose boxes to hug the actual shape edges. It doesn't need to be
pixel-perfect — consistency matters more than precision.

**Pulling corrections back out** — `cvat_export.py` writes CVAT's current
annotations for a session back to `dataset/corrected_labels/<session>/` as
plain YOLO `.txt` (one file per frame, empty file for a frame confirmed to
have no shapes):

```
python cvat_export.py --username admin --password '...' ld_1b03Gfsb   # one session
python cvat_export.py --username admin --password '...'               # every task
```

Because CVAT tracks correction state per-task rather than per-frame, this
exports every frame in a session, on the assumption you review a session
front-to-back before calling this for it. A session you haven't corrected
yet just re-exports its bootstrap boxes unchanged (harmless —
`03_prepare_dataset.py` treats it the same as any other labeled frame, just
not yet a source of improvement).

### 4. Build the dataset and train

```
python 03_prepare_dataset.py \
    --frames-dir ./dataset/raw_frames \
    --labels-dir ./dataset/corrected_labels \
    --out-dir ./dataset/yolo_dataset \
    --val-split 0.2

python 04_train_and_export.py --data ./dataset/yolo_dataset/dataset.yaml
```

`03_prepare_dataset.py` splits by session (not frame) into an Ultralytics
layout + `dataset.yaml`. `04_train_and_export.py` fine-tunes a small model
(`yolo11n.pt` by default — size matters, this runs every tracker tick during
live solving), exports to ONNX with the required contract, self-checks it,
and copies the result to `dataset/transparent_shape_nms.onnx`.

## Validating before you swap it in

Don't replace the shipped model straight from training metrics — visually
check it against the repo's existing debug harness first:

1. Temporarily point `backend/build.rs`'s `transparent_shape_model` path (or
   just overwrite `backend/resources/transparent_shape_nms.onnx` in a scratch
   branch/worktree) at your new export.
2. `cargo build`, then use the debug UI's **"Test transparent shape
   normal"/"hard"** buttons (`ui/src/debug.rs`, backed by
   `backend/src/services/debug.rs::test_transparent_shape`) to run it against
   the two canned fixture clips (`backend/resources/transparent_shape_test_{normal,hard}.mp4`).
3. Watch the `debug_shape_tracks` overlay (`backend/src/debug.rs`) — compare
   box recall (any missed shapes?) and false positives (spurious boxes can
   fool the background-direction heuristic in `shape.rs` into picking the
   wrong target) against the current model on the same clips.
4. For a more representative regression test than the two canned clips,
   consider adding 1-2 of your own held-out recordings as new fixtures the
   same way `transparent_shape_test_normal.mp4` is wired in `build.rs`
   (`resources_dir.join(...)` + `cargo:rustc-env=...`).
5. Once satisfied, replace `backend/resources/transparent_shape_nms.onnx`,
   rebuild, and do a live in-game run with `enable_transparent_shape_solving`
   on.

## Iterating

When the live solver still misses or mis-clicks, save that clip and run it
back through steps 1-2 — this time bootstrapping from your *new* model
(`--model dataset/transparent_shape_nms.onnx`) instead of the shipped one —
to mine hard examples. Correct just those in CVAT, fold them into
`dataset/corrected_labels/`, and re-run steps 4 (`03`/`04`). Compounding
targeted correction rounds like this will get you further than one big
initial labeling pass.

## Rough effort expectations

- Labeling: correcting bootstrap boxes is much faster than labeling from
  scratch — expect a few hundred to low thousands of corrected frames to
  meaningfully move accuracy, given this is a single-class fine-tune from an
  already-decent baseline, not training from zero.
- Training: `yolo11n`/`yolov8n` fine-tune on a few thousand images, single
  class, 640px — well within reach of a single consumer GPU in well under an
  hour for 100+ epochs.
