#!/usr/bin/env python3
"""
Corrected Gradio Facial Emotion Recognition app (single-file).
- Face detection: MTCNN
- Model: Keras .h5 (auto-detects model.input_shape and channels)
- Supports Image, Video, Webcam
- Batches face predictions per frame
- Includes numpy compatibility monkeypatch for older libs expecting np.bool8
- Reads optional overrides from .env:
    FER_MODEL_PATH -> path to .h5 model
    FER_LABEL_PRESET -> "affectnet_7" or "affectnet_8" (optional)
    FER_IMG_SIZE -> "W,H" (optional, overrides model input size)
"""

import os
import uuid
import json
import cv2
import numpy as np

# ---------------------- numpy compatibility monkeypatch ----------------------
# Do this early so libraries expecting older numpy aliases don't crash (quick workaround).
if not hasattr(np, "bool8"):
    # numpy >=2 removed some aliases; restore minimal alias compatibility
    np.bool8 = np.bool_
# ------------------------------------------------------------------------------

from mtcnn import MTCNN
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from packaging import version
import gradio as gr

# ----------------------- ENV & CONFIG -----------------------
load_dotenv()

# Model path and label presets
MODEL_PATH = os.environ.get("FER_MODEL_PATH", "emModel_1.h5")
LABEL_PRESET = os.environ.get("FER_LABEL_PRESET", "").strip()  # e.g. "affectnet_7"
IMG_SIZE_ENV = os.environ.get("FER_IMG_SIZE", "").strip()  # expected format "W,H" (width,height)
# NOTE: OpenCV resize takes (width, height)

# Robust labels.json location (script dir or cwd)
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
LABELS_FILE = os.path.join(BASE_DIR, "labels.json")

# Load label presets if present, else use safe defaults
if os.path.exists(LABELS_FILE):
    try:
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            LABEL_PRESETS = json.load(f)
    except Exception:
        LABEL_PRESETS = {}
else:
    LABEL_PRESETS = {}

LABEL_PRESETS.setdefault("affectnet_7", ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"])
LABEL_PRESETS.setdefault("affectnet_8", ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Anger", "Disgust", "Contempt"])

# ----------------------- GLOBALS (model will be loaded below) -----------------------
detector = MTCNN()
model = None
num_outputs = None
model_input_channels = 1
IMG_SIZE = None  # will be inferred from model or env
MODEL_STATUS = "No model loaded."

# If FER_IMG_SIZE provided in env, parse it (expecting "W,H")
if IMG_SIZE_ENV:
    try:
        parts = [int(x.strip()) for x in IMG_SIZE_ENV.split(",")]
        if len(parts) == 2:
            IMG_SIZE = (int(parts[0]), int(parts[1]))  # (W, H)
    except Exception:
        IMG_SIZE = None

# ----------------------- MODEL LOADING & AUTO-DETECT -----------------------
if MODEL_PATH and os.path.exists(MODEL_PATH):
    try:
        # load without compile to avoid compile-time metric issues
        model = load_model(MODEL_PATH, compile=False)
        num_outputs = int(model.output_shape[-1])
        inshape = model.input_shape  # e.g. (None, H, W, C)
        # Defensive parsing if model has channels_last
        if inshape is not None and len(inshape) >= 4:
            # Keras typically uses (None, H, W, C)
            h = inshape[-3]
            w = inshape[-2]
            c = inshape[-1]
            # If user hasn't set IMG_SIZE in env, infer from model
            if IMG_SIZE is None and h and w:
                IMG_SIZE = (int(w), int(h))  # (W, H)
            if c:
                model_input_channels = int(c)
        MODEL_STATUS = f"Model loaded: {MODEL_PATH} (outputs={num_outputs}, channels={model_input_channels}, img_size={IMG_SIZE})"
    except Exception as e:
        model = None
        MODEL_STATUS = f"⚠️ Could not load model '{MODEL_PATH}': {e}"
else:
    MODEL_STATUS = f"⚠️ Model file not found at: {MODEL_PATH}"

# Final fallback for IMG_SIZE
if IMG_SIZE is None:
    IMG_SIZE = (96, 96)  # default (W,H)

# choose labels
def pick_labels():
    if LABEL_PRESET and LABEL_PRESET in LABEL_PRESETS:
        return LABEL_PRESETS[LABEL_PRESET]
    if num_outputs == 7:
        return LABEL_PRESETS.get("affectnet_7")
    if num_outputs == 8:
        return LABEL_PRESETS.get("affectnet_8")
    return LABEL_PRESETS.get("affectnet_7")

EMOTION_LABELS = pick_labels() or LABEL_PRESETS["affectnet_7"]

# ----------------------- HELPERS -----------------------
def _clip_box(x, y, w, h, W, H):
    x = max(0, x)
    y = max(0, y)
    w = max(1, w)
    h = max(1, h)
    if x + w > W:
        w = W - x
    if y + h > H:
        h = H - y
    return x, y, w, h

def _prepare_face(face_img_bgr):
    """
    Prepare a single face crop to the model's expected shape.
    IMG_SIZE is (W,H) — OpenCV resize expects (W,H) as dsize.
    Returns (1,H,W,C) float32 array normalized to [0,1].
    """
    if model_input_channels == 1:
        gray = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
        arr = resized.astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=-1)  # (H,W,1)
    else:
        rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
        arr = resized.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1,H,W,C)
    return arr

def _prepare_faces_batch(crops_bgr_list):
    """
    Accepts list of BGR face crops; returns numpy array (N,H,W,C) for model.
    """
    if len(crops_bgr_list) == 0:
        return np.zeros((0, IMG_SIZE[1], IMG_SIZE[0], model_input_channels), dtype="float32")
    prepared = [np.squeeze(_prepare_face(crop)) for crop in crops_bgr_list]
    batch = np.stack(prepared, axis=0).astype("float32")
    # if grayscale produced (N,H,W) expand channel axis
    if batch.ndim == 3:
        batch = np.expand_dims(batch, axis=-1)
    return batch

def _predict_batch(crops_bgr_list):
    """
    Predict a batch of crops (list). Returns list of (label, confidence).
    If model absent, returns (None, 0.0) for each item.
    """
    if model is None or len(crops_bgr_list) == 0:
        return [(None, 0.0)] * len(crops_bgr_list)
    batch = _prepare_faces_batch(crops_bgr_list)
    # defensive shape check
    expected = model.input_shape  # (None, H, W, C) or similar
    if expected is not None and len(expected) >= 4:
        expected_dims = tuple(int(x) if x is not None else None for x in expected[1:])
        actual_dims = tuple(batch.shape[1:])
        if expected_dims != actual_dims:
            raise ValueError(
                f"Input batch dimensions {actual_dims} do not match model expected {expected_dims}. "
                f"Set FER_IMG_SIZE env var to match model or re-train the model for {actual_dims}."
            )
    try:
        preds = model.predict(batch, verbose=0)
    except Exception as e:
        # Surface clear error
        raise RuntimeError(f"Error during model.predict: {e}")
    out = []
    for p in preds:
        idx = int(np.argmax(p))
        conf = float(p[idx])
        label = EMOTION_LABELS[idx] if idx < len(EMOTION_LABELS) else f"Class {idx}"
        out.append((label, conf))
    return out

def _annotate(image_bgr, box, label, conf):
    x, y, w, h = box
    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
    txt = f"{label} ({conf*100:.1f}%)"
    ty = max(12, y - 8)
    cv2.putText(image_bgr, txt, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

# ----------------------- CORE -----------------------
def predict_emotion(image_np):
    """
    Process a single image (numpy RGB from Gradio) and return annotated RGB image
    """
    if image_np is None:
        return None
    image_rgb = image_np.copy()
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    H, W = image_bgr.shape[:2]

    faces = detector.detect_faces(image_rgb)
    crops = []
    boxes = []
    for f in faces:
        x, y, w, h = f.get("box", [0, 0, 0, 0])
        x, y, w, h = _clip_box(int(x), int(y), int(w), int(h), W, H)
        face_crop = image_bgr[y:y + h, x:x + w]
        if face_crop.size == 0:
            continue
        crops.append(face_crop)
        boxes.append((x, y, w, h))

    if not crops:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    preds = _predict_batch(crops)
    for (label, conf), box in zip(preds, boxes):
        if label is None:
            continue
        _annotate(image_bgr, box, label, conf)

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

def process_video(video_path):
    """Process uploaded video file path, annotate frames, and write out mp4 path"""
    if not video_path:
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_name = f"processed_{uuid.uuid4().hex}.mp4"
    out_path = os.path.join(os.getcwd(), out_name)
    # mp4v is commonly available; change codec if needed on your platform
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(frame_rgb)

        crops = []
        boxes = []
        for f in faces:
            x, y, w, h = f.get("box", [0, 0, 0, 0])
            x, y, w, h = _clip_box(int(x), int(y), int(w), int(h), W, H)
            crop = frame_bgr[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            crops.append(crop)
            boxes.append((x, y, w, h))

        if crops:
            preds = _predict_batch(crops)
            for (label, conf), box in zip(preds, boxes):
                if label is None:
                    continue
                _annotate(frame_bgr, box, label, conf)

        writer.write(frame_bgr)

    cap.release()
    writer.release()
    return out_path

def webcam_frame(frame):
    # frame is numpy RGB from gradio webcam
    return predict_emotion(frame)

# ----------------------- UI -----------------------
def build_ui():
    # pick webcam input API based on gradio version
    if version.parse(gr.__version__) < version.parse("3.37.0"):
        webcam_input = gr.Image(source="webcam", streaming=True, type="numpy", label="Webcam")
    else:
        webcam_input = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam")

    image_iface = gr.Interface(
        fn=predict_emotion,
        inputs=gr.Image(type="numpy", label="Upload Image"),
        outputs=gr.Image(type="numpy", label="Annotated Image"),
        title="Facial Emotion Recognition (AffectNet)",
        description=f"Upload an image to detect emotions. {MODEL_STATUS}"
    )

    video_iface = gr.Interface(
        fn=process_video,
        inputs=gr.Video(label="Upload Video"),
        outputs=gr.Video(label="Processed Video"),
        title="Video Emotion Recognition",
        description="Upload a video; the app outputs an annotated MP4."
    )

    webcam_iface = gr.Interface(
        fn=webcam_frame,
        inputs=webcam_input,
        outputs=gr.Image(type="numpy", label="Live Output"),
        live=True,
        title="Webcam Emotion Recognition",
        description="Live detection via your webcam."
    )

    return gr.TabbedInterface([image_iface, video_iface, webcam_iface],
                              tab_names=["Image Input", "Video Input", "Webcam"])

# ----------------------- ENTRYPOINT -----------------------
if __name__ == "__main__":
    print("Starting FER app")
    print(MODEL_STATUS)
    print(f"Using IMG_SIZE (W,H) = {IMG_SIZE}, model_input_channels = {model_input_channels}")
    ui = build_ui()
    # share=False by default; set True if you want a public link
    ui.launch(share=False, inbrowser=True)
