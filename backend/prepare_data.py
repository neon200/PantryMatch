import pandas as pd
import re

# 1. Load the CSV file from the data folder
df = pd.read_csv("backend/data/Cleaned_Indian_Food_Dataset.csv")

# 2. Select relevant columns
# Adjust column names if needed after first run
df = df[['TranslatedRecipeName', 'Cleaned-Ingredients', 'TotalTimeInMins', 'TranslatedInstructions', 'Cuisine']]

# 3. Define text cleaning function
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Keep commas for separating ingredients; remove other special characters
    text = re.sub(r'[^a-z0-9, ]', '', text)
    return text

# 4. Apply cleaning function on ingredients column
df['processed_ingredients'] = df['Cleaned-Ingredients'].apply(clean_text)

# 5. Save cleaned CSV file for future fast loading in your app
df.to_csv("backend/data/final_recipes.csv", index=False)

print(f"Success! Saved {len(df)} recipes to 'final_recipes.csv'")
print("Sample processed ingredients:", df['processed_ingredients'].iloc[0])
