from ml_infer_ingredients import load_model, predict_ingredients_from_bytes

model, class_names, device = load_model()
with open("backend/data/Train/Tomato/Tomato_1.jpg", "rb") as f:
    img_bytes = f.read()

preds = predict_ingredients_from_bytes(model, class_names, device, img_bytes, top_k=5)
print(preds)