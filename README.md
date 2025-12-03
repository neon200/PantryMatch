# 🍳 PantryMatch

> **Cook smarter, not harder** - Find delicious recipes from your pantry ingredients using AI-powered matching and computer vision

PantryMatch is an intelligent recipe discovery platform that helps you find the perfect recipes based on ingredients you already have. It uses machine learning (TF-IDF and cosine similarity) to match your pantry items with recipes, **computer vision (ResNet18 CNN)** to detect ingredients from photos, **AI vision models** for enhanced detection, and AI to suggest ingredient substitutions when you're missing something.

![PantryMatch](https://img.shields.io/badge/PantryMatch-Recipe%20Finder-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![React](https://img.shields.io/badge/React-19.2-blue?style=for-the-badge&logo=react)
![Flask](https://img.shields.io/badge/Flask-3.1-green?style=for-the-badge&logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?style=for-the-badge&logo=pytorch)

## ✨ Features

- 🔍 **Smart Recipe Search** - Enter your ingredients and get matched recipes using TF-IDF vectorization and cosine similarity
- 📊 **Match Score** - See how well each recipe matches your ingredients (0-100%)
- 📸 **Image-Based Ingredient Detection** - Two powerful options:
  - **Option 1**: Upload separate images of individual ingredients (uses ResNet18 CNN + AI vision)
  - **Option 2**: Upload a single combined image with all ingredients (uses AI vision only)
- 🤖 **AI Ingredient Substitution** - Get intelligent suggestions when you're missing an ingredient
- 🎥 **Video Tutorials** - Access YouTube video tutorials for each recipe
- 🧠 **Hybrid Detection System** - Combines custom-trained ResNet18 model with OpenRouter vision API for best accuracy
- 🎨 **Beautiful UI** - Modern, food-themed design with warm colors and smooth animations
- ⚡ **Fast & Responsive** - Optimized search with pre-processed recipe data

## 🛠️ Tech Stack

### Backend
- **Python 3.11+**
- **Flask** - Web framework
- **PyTorch** - Deep learning framework for ResNet18 model
- **scikit-learn** - TF-IDF vectorization and cosine similarity
- **pandas** - Data processing
- **Pillow (PIL)** - Image processing
- **OpenRouter API** - AI-powered ingredient substitution and vision (GPT-4o-mini)
- **RapidAPI** - YouTube video search

### Frontend
- **React 19.2** - UI framework
- **Vite** - Build tool
- **CSS3** - Custom styling with modern design

### Machine Learning
- **ResNet18** - Pre-trained CNN architecture fine-tuned on 51 ingredient classes
- **Transfer Learning** - Fine-tuning pre-trained ResNet18 for ingredient classification
- **Custom Dataset** - 51 classes of fruits and vegetables (Train/val split)

## 📋 Prerequisites

- Python 3.11 or higher
- Node.js 18+ and npm
- API Keys:
  - OpenRouter API key (for AI substitutions and vision)
  - RapidAPI key (for YouTube videos)
- GPU (optional but recommended for training the CNN model)

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/PantryMatch.git
cd PantryMatch
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install flask flask-cors pandas scikit-learn requests torch torchvision pillow

# Prepare recipe data
python prepare_data.py

# Set up API keys
# Copy the example config file and add your API keys
cp config.example.py config.py
# Then edit config.py with your actual API keys
```

### 3. Train the Ingredient Classification Model (Optional)

If you want to train your own ResNet18 model or retrain with new data:

```bash
cd backend

# Make sure you have the dataset structure:
# backend/data/Train/ (with subdirectories for each ingredient class)
# backend/data/val/ (with subdirectories for each ingredient class)

# Train the model
python ml_train_ingredients_model.py

# The trained model will be saved to:
# backend/models/ingredients_resnet18.pt
```

**Note**: Training requires a dataset organized as:
```
backend/data/
├── Train/
│   ├── Apple/
│   │   ├── Apple_1.jpg
│   │   ├── Apple_2.jpg
│   │   └── ...
│   ├── Banana/
│   └── ... (other ingredient classes)
└── val/
    ├── Apple/
    ├── Banana/
    └── ... (validation images)
```

The model will automatically detect the number of classes from the directory structure.

### 4. Frontend Setup

```bash
cd frontend/frontend

# Install dependencies
npm install
```

## 🎯 Usage

### Start Backend Server

```bash
cd backend
python app.py
```

The Flask server will run on `http://127.0.0.1:5000`

**Note**: On first run, the ResNet18 model will be loaded (this may take a few seconds). The model file should be at `backend/models/ingredients_resnet18.pt`.

### Start Frontend Development Server

```bash
cd frontend/frontend
npm run dev
```

The React app will run on `http://localhost:5173` (or another port if 5173 is busy)

### Using the Application

1. **Search Recipes**: Enter your ingredients (comma-separated) in the search box
2. **Image Detection** (Two Options):
   - **Option 1 - Separate Images**: Upload multiple images, one per ingredient. The ResNet18 model analyzes each image, and AI vision refines the results. See "Detected by model" chips for CNN predictions.
   - **Option 2 - Combined Image**: Upload a single image containing all ingredients. Uses AI vision directly for detection.
3. **View Results**: Browse matched recipes with match scores
4. **View Recipe Details**: Click "View Recipe" to see full instructions
5. **Get Substitutions**: Enter a missing ingredient to get AI-powered suggestions
6. **Watch Tutorials**: Access YouTube video tutorials for visual guidance

## 📡 API Endpoints

### `GET /search`
Search for recipes based on ingredients.

**Query Parameters:**
- `q` (string): Comma-separated list of ingredients

**Response:**
```json
[
  {
    "title": "Recipe Name",
    "ingredients": "ingredient1, ingredient2, ...",
    "instructions": "Step-by-step instructions...",
    "time": 30,
    "score": 0.85
  }
]
```

### `POST /classify-image`
Detect ingredients from uploaded image(s). Supports two modes via query parameter.

**Query Parameters:**
- `mode` (string, optional): 
  - `cnn` (default): Uses ResNet18 + optional OpenRouter vision
  - `llm_only`: Skips ResNet18, uses only OpenRouter vision

**Request Body:**
- `multipart/form-data` with:
  - `image` (file): Single image file (for `mode=llm_only`)
  - `images` (files): Multiple image files (for `mode=cnn`)

**Response:**
```json
{
  "ingredients": ["chicken", "apple", "corn", "cabbage"],
  "cnn_ingredients": ["Cabbage", "Corn", "Apple"],
  "llm_ingredients": ["chicken", "apple", "corn", "cabbage", "salt"],
  "per_image_predictions": [
    {
      "filename": "image1.jpg",
      "predictions": [
        {"name": "Cabbage", "prob": 0.996},
        {"name": "Coconut", "prob": 0.001}
      ]
    }
  ]
}
```

**Response Fields:**
- `ingredients`: Final merged list (prefers LLM if available, otherwise CNN)
- `cnn_ingredients`: Ingredients detected by ResNet18 model only
- `llm_ingredients`: Ingredients detected by OpenRouter vision API
- `per_image_predictions`: Detailed predictions per image with probabilities

### `POST /adapt`
Get AI-powered ingredient substitution suggestions.

**Request Body:**
```json
{
  "title": "Recipe Name",
  "instructions": "Recipe instructions...",
  "missing": "missing ingredient"
}
```

**Response:**
```json
{
  "adaptedStep": "AI-generated substitution suggestion..."
}
```

### `GET /videos`
Get YouTube video tutorials for a recipe.

**Query Parameters:**
- `recipe` (string): Recipe name

**Response:**
```json
[
  {
    "title": "Video Title",
    "url": "https://youtube.com/watch?v=..."
  }
]
```

## 📁 Project Structure

```
PantryMatch/
├── backend/
│   ├── app.py                      # Flask application with all endpoints
│   ├── prepare_data.py            # Data preprocessing script
│   ├── ml_train_ingredients_model.py  # ResNet18 training script
│   ├── ml_infer_ingredients.py    # Model loading and inference helpers
│   ├── config.py                  # API keys (not in git)
│   ├── config.example.py          # API keys template
│   ├── data/
│   │   ├── Cleaned_Indian_Food_Dataset.csv
│   │   ├── final_recipes.csv      # Processed recipe data
│   │   ├── Train/                 # Training images (51 classes)
│   │   │   ├── Apple/
│   │   │   ├── Banana/
│   │   │   └── ... (other classes)
│   │   └── val/                   # Validation images
│   │       ├── Apple/
│   │       ├── Banana/
│   │       └── ... (other classes)
│   └── models/
│       ├── ingredients_resnet18.pt  # Trained ResNet18 model
│       └── ingredients_classes.txt   # Class names list
│
└── frontend/
    └── frontend/
        ├── src/
        │   ├── App.jsx            # Main React component
        │   ├── App.css            # Component styles
        │   └── index.css          # Global styles
        ├── package.json
        └── vite.config.js
```

## 🔬 How It Works

### Recipe Matching Algorithm

1. **Data Preprocessing**: Recipe ingredients are cleaned and normalized
2. **TF-IDF Vectorization**: Both user query and recipes are converted to TF-IDF vectors
3. **Cosine Similarity**: Computes similarity between query and each recipe
4. **Ranking**: Recipes are sorted by match score (0-100%)

**Formula:**
```
Match Score = cosine_similarity(user_ingredients, recipe_ingredients) × 100
```

### Image-Based Ingredient Detection

PantryMatch uses a **hybrid approach** combining custom ML and AI vision:

#### Option 1: Separate Images (CNN + AI Vision)
1. **ResNet18 Analysis**: Each uploaded image is analyzed by a custom-trained ResNet18 model
2. **Confidence Filtering**: Only predictions above 0.5 confidence are kept
3. **AI Vision Enhancement**: All images are sent to OpenRouter's GPT-4o-mini vision model
4. **Result Merging**: CNN and AI vision results are combined and deduplicated
5. **Display**: CNN-only detections shown separately; final list uses AI vision when available

#### Option 2: Combined Image (AI Vision Only)
1. **Direct AI Analysis**: Single image sent directly to OpenRouter vision API
2. **Ingredient Extraction**: AI model identifies all visible ingredients
3. **Result**: Clean ingredient list ready for recipe search

**Model Architecture:**
- **Base**: ResNet18 (pre-trained on ImageNet)
- **Fine-tuning**: Last fully-connected layer replaced for 51-class classification
- **Training**: Transfer learning with Adam optimizer, learning rate scheduling
- **Classes**: 51 ingredient types (fruits and vegetables)

### AI Substitution

When a user is missing an ingredient, the app:
1. Sends the recipe and missing ingredient to OpenRouter API (Grok model)
2. Gets context-aware substitution suggestions
3. Provides Indian cooking-specific alternatives when applicable

## 🎨 Design Philosophy

The UI features a warm, food-themed design with:
- Orange and warm color palette
- Clean, modern layout
- Smooth animations and transitions
- Responsive design for all devices
- Accessible focus states and keyboard navigation
- Clear separation between CNN and AI vision detections

## 🧪 Model Training Details

### Training Configuration
- **Architecture**: ResNet18
- **Input Size**: 224x224 pixels
- **Batch Size**: 32
- **Epochs**: 20 (with early stopping)
- **Optimizer**: Adam (lr=0.001)
- **Scheduler**: ReduceLROnPlateau
- **Loss Function**: CrossEntropyLoss
- **Data Augmentation**: Random transforms (rotation, flip, color jitter)

### Model Performance
- The model is trained on a dataset of 51 ingredient classes
- Validation accuracy is monitored during training
- Best model checkpoint is saved based on validation performance

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Recipe dataset: Indian Food Dataset
- AI Model: Grok & GPT-4o-mini via OpenRouter
- Video API: RapidAPI YouTube Alternative
- Deep Learning Framework: PyTorch
- Pre-trained Model: ResNet18 (ImageNet)

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Made with ❤️ for food lovers who want to cook smarter**
