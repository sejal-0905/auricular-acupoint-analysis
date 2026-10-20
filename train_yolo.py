"""
Trains the YOLO pose model on your real dataset (~50 images).

Compared to the 4-photo practice run: more epochs, since real learning takes
longer with more (though still modest) data. Still don't expect great accuracy
at 50 images - this is a meaningful step up from 4, but the field's benchmark
paper used 660. Treat this as "does accuracy start moving off zero" - it should,
but likely won't be production-quality yet.

Usage:
    python train_yolo.py
"""

from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")

results = model.train(
    data="data.yaml",
    epochs=150,       # more than the practice run's 50, since there's more to learn
    imgsz=640,
    batch=8,           # can afford a slightly bigger batch with more images
    patience=30,       # stops early if it stops improving for 30 epochs straight
    project="../models",
    name="ear_acupoint_v2_run",
)

print("\nTraining complete. Check ../models/ear_acupoint_v2_run/ for results.")
print("Open results.png to see loss curves and accuracy over training.")
print("\nKey thing to check this time: metrics/mAP50(P) in results.png (bottom-right area)")
print("should show SOME upward movement now, unlike the flat zero from the 4-photo run.")
