import google.generativeai as genai

genai.configure(api_key="AIzaSyCfTY2Uglf4p-4N07KsJqbk89tZS6SBXYE")

for m in genai.list_models():
    print(m)
