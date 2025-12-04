from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import os
import base64

from ml_infer_ingredients import load_model, predict_ingredients_from_bytes

# -- Import API Keys from config --
try:
    from config import (
        OPENROUTER_API_KEY,
        OPENROUTER_API_URL,
        OPENROUTER_MODEL,
        RAPIDAPI_KEY,
        RAPIDAPI_HOST
    )
except ImportError:
    # Fallback to environment variables if config.py doesn't exist
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
    OPENROUTER_API_URL = os.getenv('OPENROUTER_API_URL', 'https://openrouter.ai/api/v1/chat/completions')
    OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'openai/gpt-4o-mini')
    RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY', '')
    RAPIDAPI_HOST = os.getenv('RAPIDAPI_HOST', 'youtube-v3-alternative.p.rapidapi.com')
    
    if not OPENROUTER_API_KEY or not RAPIDAPI_KEY:
        print("Warning: API keys not found. Please create config.py or set environment variables.")

app = Flask(__name__)
CORS(app)

# -- Load Recipes & TF-IDF Search Setup --
df = pd.read_csv("data/final_recipes.csv")
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['processed_ingredients'])

# -- 1. Recipe Search Endpoint --
@app.route('/search', methods=['GET'])
def search():
    user_query = request.args.get('q', '')
    user_vec = vectorizer.transform([user_query])
    scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    # Get top 6 most relevant recipes instead of 5
    top_indices = scores.argsort()[-6:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            'title': str(df.iloc[idx]['TranslatedRecipeName']),
            'ingredients': str(df.iloc[idx]['processed_ingredients']),
            'instructions': str(df.iloc[idx]['TranslatedInstructions']),
            'time': int(df.iloc[idx]['TotalTimeInMins']) if pd.notnull(df.iloc[idx]['TotalTimeInMins']) else None,
            'score': float(scores[idx])
        })
    return jsonify(results)

# -- 2. AI Adaptation Endpoint (OpenRouter) --
@app.route('/adapt', methods=['POST'])
def adapt():
    data = request.get_json()
    instructions = data.get('instructions', '')
    missing_ingredient = data.get('missing', '')
    title = data.get('title', '')
    prompt = (
        f"You are a smart Indian cooking assistant. Here is the recipe for '{title}'.\n"
        f"Instructions:\n{instructions}\n"
        f"The user does NOT have '{missing_ingredient}'. Suggest the best adaptation or workaround step (with local Indian knowledge), "
        f"explaining briefly what to do. Only output the modified or additional instruction, not the whole recipe."
    )
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful Indian recipe assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "PantryMatch adaptation"
    }
    try:
        resp = requests.post(OPENROUTER_API_URL, headers=headers, data=json.dumps(payload), timeout=40)
        if resp.status_code == 200:
            reply = resp.json()["choices"][0]["message"]["content"]
            return jsonify({"adaptedStep": reply})
        else:
            error_message = resp.text
            return jsonify({"adaptedStep": f"Could not fetch adaptation. Error: {error_message}"}), resp.status_code
    except Exception as e:
        return jsonify({"adaptedStep": "Could not fetch adaptation. Error: " + str(e)}), 500

# -- 3. YouTube Video Search Endpoint (RapidAPI, robust ID fix) --
@app.route('/videos', methods=['GET'])
def get_youtube_videos():
    recipe = request.args.get('recipe', '')
    # Only return the top 2 most relevant videos
    max_results = 2
    url = f"https://youtube-v3-alternative.p.rapidapi.com/search"
    params = {"query": recipe + " recipe", "maxResults": str(max_results)}
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST
    }
    resp = requests.get(url, headers=headers, params=params, timeout=20)
    print("RapidAPI response:", resp.status_code, resp.text)  # Debug

    results_key = "results"
    try:
        data = resp.json()
        if results_key not in data:
            if "data" in data:
                results_key = "data"
            elif "items" in data:
                results_key = "items"
        entries = data.get(results_key, [])
    except Exception as e:
        entries = []
        print("JSON parse error:", str(e))

    videos = []
    for item in entries[:max_results]:
        # Try link, then videoId/id, fallback empty string if not found
        link = item.get("link")
        if not link or link == "https://youtube.com/watch?v=None":
            video_id = item.get("id") or item.get("videoId")
            if video_id and video_id != "None":
                link = f"https://youtube.com/watch?v={video_id}"
            else:
                link = ""
        videos.append({
            "title": item.get("title", "No Title"),
            "url": link
        })

    if resp.status_code == 429:
        return jsonify({"error": "RapidAPI quota exceeded"}), 429
    elif resp.status_code != 200:
        return jsonify({"error": resp.text}), resp.status_code
    return jsonify(videos)

# -- 4. Image Ingredient Classification Endpoint (ResNet18) --
ING_MODEL = None
ING_CLASSES = None
ING_DEVICE = None


def get_ingredient_model():
    """
    Lazy-load the trained ingredient classification model.
    """
    global ING_MODEL, ING_CLASSES, ING_DEVICE
    if ING_MODEL is None:
        print("Loading ingredient classification model...")
        model, class_names, device = load_model()
        ING_MODEL = model
        ING_CLASSES = class_names
        ING_DEVICE = device
    return ING_MODEL, ING_CLASSES, ING_DEVICE


@app.route('/classify-image', methods=['POST'])
def classify_image():
    """
    Classify ingredients from an uploaded image using the trained ResNet model.

    Accepts:
      - multipart/form-data with 'image' file

    Returns:
      - ingredients: list of ingredient names above a confidence threshold
      - predictions: full top-k predictions with probabilities
    """
    # Mode:
    #   - 'cnn' (default): use ResNet + optional OpenRouter
    #   - 'llm_only': skip ResNet, only call OpenRouter on uploaded image(s)
    mode = request.args.get('mode', 'cnn')

    # Support both single file ('image') and multiple files ('images')
    files = []
    if 'images' in request.files:
        files = request.files.getlist('images')
    elif 'image' in request.files:
        files = [request.files['image']]

    files = [f for f in files if f and f.filename]
    if not files:
        return jsonify({"error": "No image file provided"}), 400

    try:
        # Read all image bytes once
        image_bytes_list = [f.read() for f in files]

        all_preds = []
        cnn_ingredients_unique = []
        extra_ingredients = []

        # -------- Mode: CNN (multi images, ResNet + optional LLM) --------
        if mode == 'cnn':
            model, class_names, device = get_ingredient_model()
            threshold = 0.5  # confident CNN predictions

            cnn_ingredients = []
            for f, img_bytes in zip(files, image_bytes_list):
                preds = predict_ingredients_from_bytes(
                    model, class_names, device, img_bytes, top_k=10
                )
                all_preds.append({"filename": f.filename, "predictions": preds})
                for p in preds:
                    if p["prob"] >= threshold:
                        cnn_ingredients.append(p["name"])

            # Deduplicate CNN ingredients
            seen_cnn = set()
            for name in cnn_ingredients:
                key = name.lower()
                if key not in seen_cnn:
                    seen_cnn.add(key)
                    cnn_ingredients_unique.append(name)

        # -------- Shared LLM (OpenRouter) for both modes --------
        try:
            if image_bytes_list and OPENROUTER_API_KEY:
                if mode == 'llm_only':
                    # Single-image, LLM‑only path: use first image only
                    img_b64 = base64.b64encode(image_bytes_list[0]).decode("utf-8")
                    data_url = f"data:image/jpeg;base64,{img_b64}"
                    content_parts = [
                        {
                            "type": "text",
                            "text": (
                                "Look at this image and list all food ingredients you can see. "
                                "Return ONLY a comma-separated list of ingredient names, "
                                "no explanations or extra text."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ]
                else:
                    # CNN mode: send all images and hint with known CNN ingredients
                    known_list = ", ".join(cnn_ingredients_unique) if cnn_ingredients_unique else "none"
                    content_parts = [
                        {
                            "type": "text",
                            "text": (
                                "Look at these images and list all food ingredients you can see. "
                                "Return ONLY a comma-separated list of ingredient names, "
                                "no explanations or extra text.\n\n"
                                f"Ingredients already detected by another model: {known_list}.\n"
                                "Use that list as a hint but still include every ingredient you see in the final response, even if it was already detected."
                            ),
                        }
                    ]
                    for img_bytes in image_bytes_list:
                        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                        data_url = f"data:image/jpeg;base64,{img_b64}"
                        content_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            }
                        )

                vision_payload = {
                    "model": OPENROUTER_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a vision model that identifies visible food ingredients from images.",
                        },
                        {
                            "role": "user",
                            "content": content_parts,
                        },
                    ],
                    "max_tokens": 256,
                }

                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                }

                vision_resp = requests.post(
                    OPENROUTER_API_URL, headers=headers, data=json.dumps(vision_payload), timeout=40
                )

                print("OpenRouter vision status:", vision_resp.status_code)

                try:
                    resp_json = vision_resp.json()
                except Exception:
                    print("OpenRouter vision non-JSON response:", vision_resp.text[:500])
                    resp_json = {}

                if vision_resp.status_code == 200 and "choices" in resp_json:
                    content = resp_json["choices"][0]["message"]["content"].strip()
                    for raw in content.replace("\n", ",").split(","):
                        name = raw.strip(" -•\t")
                        if name:
                            extra_ingredients.append(name)
                    print("OpenRouter vision ingredients:", extra_ingredients)
                else:
                    if resp_json or vision_resp.text:
                        print("OpenRouter vision error body:", resp_json or vision_resp.text[:500])
        except Exception as vision_err:
            print("Hybrid OpenRouter vision call failed:", str(vision_err))

        # Final ingredients:
        #  - If LLM returned anything, use that list.
        #  - Else in CNN mode, fall back to CNN ingredients.
        all_ings = []
        seen = set()
        if extra_ingredients:
            source_list = extra_ingredients
        elif mode == 'cnn':
            source_list = cnn_ingredients_unique
        else:
            source_list = []

        for name in source_list:
            key = name.lower()
            if key not in seen:
                seen.add(key)
                all_ings.append(name)

        return jsonify({
            "ingredients": all_ings,
            "cnn_ingredients": cnn_ingredients_unique,
            "llm_ingredients": extra_ingredients,
            "per_image_predictions": all_preds,
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        print("Error in /classify-image:", str(e))
        return jsonify({"error": "Failed to classify image", "details": str(e)}), 500


# -- Run App --
if __name__ == '__main__':
    app.run(debug=True)
