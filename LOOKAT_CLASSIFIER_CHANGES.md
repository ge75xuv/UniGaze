# LookAt/NotLookAt Classifier - Added Components

## Overview
This document details the additions made to `unigaze/predict_gaze_video.py` to implement automatic LookAt/NotLookAt classification for each frame.

---

## 1. **CLI Parameters** (Lines ~106-125)
### Purpose
Expose classifier thresholds and settings as command-line arguments for easy tuning.

### Marked with: `# === ADDED: LookAt/NotLookAt Classifier Parameters ===`

**New arguments:**
- `--lookat_angle_enter` (default: 8.0°) — Angle threshold to enter LookAt state
- `--lookat_angle_exit` (default: 12.0°) — Angle threshold to exit LookAt state (hysteresis)
- `--lookat_smooth_window` (default: 5) — Moving average window size for angle smoothing
- `--lookat_ref_frames` (default: 3) — Number of initial frames to build reference gaze
- `--lookat_hold_missed` (default: 3) — Frames to retain label during face detection failure
- `--draw_angle` (default: True) — Whether to draw angle overlay on video for debugging

---

## 2. **Classifier State Initialization** (Lines ~241-257)
### Purpose
Initialize per-video state before processing frames, including reference gaze and label history.

### Marked with: `# === ADDED: LookAt Classifier Initialization ===`

**Initialized variables:**
- `reference_gaze_vec` — Normalized 3D gaze direction from first frames (None until built)
- `reference_buffer` — Accumulates gaze vectors from initial frames
- `angle_history` — Rolling history of angular distances to reference
- `is_lookat` — Current classification state (initialized True)
- `missed_counter` — Tracks consecutive frames with failed face detection

---

## 3. **Per-Frame State Variables** (Lines ~274-276)
### Purpose
Reset per-frame classification flags before processing each frame.

### Marked with: `# === ADDED: Per-frame classifier state variables ===`

**Frame-local variables:**
- `has_valid_gaze` — Flag indicating if gaze was successfully estimated
- `frame_angle_smooth` — Smoothed angular distance (None if no valid gaze)

---

## 4. **Gaze Vector Recording** (Lines ~282-283)
### Purpose
Store the denormalized 3D gaze vectors for each detected face for classification.

### Marked with: `# === ADDED: Dictionary to store denormalized gaze vectors ===`

**New data structure:**
- `gaze_vec_record[idx]` — Maps face index → 3D gaze vector in camera coordinates

---

## 5. **Gaze Vector Capture** (Line ~373)
### Purpose
Extract and save the denormalized gaze vector for each face after gaze estimation.

### Marked with: `# === ADDED: Store denormalized 3D gaze vector ===`

Captures `pred_gaze_cancel_nor` (3D gaze in camera coordinates) for later classification.

---

## 6. **LookAt Classification Logic** (Lines ~413-453)
### Purpose
Core classifier: builds reference, computes angular distance, applies hysteresis thresholds.

### Marked with: `# === ADDED: LookAt/NotLookAt Classification Logic ===`

**Three stages:**

#### **Stage 1: Build Reference (first frames)**
```
if reference_gaze_vec is None:
    - Buffer gaze vectors from first ref_frames valid detections
    - Average and normalize to create stable reference
    - Set is_lookat = True (first frame assumption)
```

#### **Stage 2: Classify by Angular Distance**
```
- Compute dot product between current and reference gaze
- Convert to angular distance (degrees)
- Apply moving average smoothing over smooth_window frames
```

#### **Stage 3: Hysteresis Thresholds**
```
if is_lookat:
    - Exit LookAt only if angle > angle_exit
else:
    - Enter LookAt only if angle < angle_enter
```

This prevents flickering near the decision boundary.

---

## 7. **Handle Face Detection Failures** (Lines ~455-461)
### Purpose
Maintain label stability when face is temporarily not detected.

### Marked with: `# === ADDED: Handle frames where face detection failed ===`

**Logic:**
- Increment `missed_counter` on failed detection
- After `hold_missed` consecutive failures, switch to NotLookAt
- Reset counter on successful detection

---

## 8. **Visualize Classification Result** (Lines ~463-478)
### Purpose
Draw LookAt/NotLookAt label directly on output frames in real time.

### Marked with: `# === ADDED: Visualize LookAt/NotLookAt label on video frame ===`

**Visualization:**
- **Label text** (top-left, size 1.2):
  - Green (60, 210, 60) for "LookAt"
  - Red (40, 60, 235) for "NotLookAt"
- **Optional angle debug overlay** (below label):
  - Shows smoothed angle in degrees
  - Useful for threshold tuning
  - Toggleable via `--draw_angle`

---

## Example Usage

```bash
python unigaze/predict_gaze_video.py \
  -i /path/to/video.mp4 \
  -out /path/to/output \
  -m unigaze/configs/config.yaml \
  --ckpt_resume /path/to/checkpoint.pt \
  --lookat_ref_frames 5 \
  --lookat_angle_enter 7 \
  --lookat_angle_exit 13 \
  --lookat_smooth_window 7 \
  --draw_angle True
```

---

## Tuning Guide

1. **Angle thresholds** (`angle_enter`, `angle_exit`):
   - Start at 8–12 degrees
   - **Lower** → more sensitive to gaze changes
   - **Higher** → more tolerant

2. **Smoothing window** (`smooth_window`):
   - Larger → more stable, slower response
   - Smaller → faster response, more noise

3. **Reference frames** (`ref_frames`):
   - More frames → more robust reference
   - Fewer frames → quicker initialization

4. **Missed frame tolerance** (`hold_missed`):
   - Higher → tolerates brief detection gaps
   - Lower → switches to NotLookAt faster on loss of face

**Tip:** Use `--draw_angle True` to visualize angle over time, then adjust thresholds in a second pass.

---

## Implementation Notes

- **Selected face:** Uses the **largest detected face** each frame as the subject of classification
- **Gaze reference:** Built from first N valid frames, then remains fixed for entire video
- **Angular distance:** Computed as `arccos(dot_product)` between normalized 3D gaze vectors
- **Output video:** All frames written to MP4 with labels, unaffected by `save_freq`
- **Image saving:** Original image-save frequency (`save_freq=30`) unaffected by classification
