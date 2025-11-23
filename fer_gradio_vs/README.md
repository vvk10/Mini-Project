# Facial Emotion Recognition (Gradio • AffectNet)

A complete Gradio app that detects faces with **MTCNN** and predicts emotions using your trained **Keras `.h5`** model (AffectNet 7- or 8-class variants).

## Quick start

1. Create a virtual environment and install deps:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy your trained model into this folder and name it `Emodel.h5` (or edit `.env`).
3. (Optional) copy `.env.example` to `.env` and adjust:
   ```bash
   FER_MODEL_PATH=Emodel.h5
   FER_LABEL_PRESET=affectnet_8   # or affectnet_7
   FER_IMG_SIZE=96,96
   ```
4. Run:
   ```bash
   python app.py
   ```
   or in VS Code: **Run and Debug → Run Gradio FER**.

## Tabs
- **Image Input**: returns an annotated image.
- **Video Input**: returns an annotated mp4 saved to `/mnt/data` on this environment or to the project folder locally.
- **Webcam**: live stream prediction.

If your model uses 8 classes (includes **Contempt**), set `FER_LABEL_PRESET=affectnet_8`. If not set, the app infers 7 vs 8 classes from the model's last layer.
