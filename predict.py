from ultralytics import YOLO

MODEL_PATH = "runs/models/ear_acupoint_v2_run-3/weights/best.pt"
IMAGE_PATH = "../data/images/HE_10_R.jpg"  

model = YOLO(MODEL_PATH)
results = model(IMAGE_PATH)

output_path = "test_prediction.jpg"
results[0].save(output_path)

print(f"\nDone. Open '{output_path}' to see the prediction.")