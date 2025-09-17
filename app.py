from flask import Flask, request, jsonify
import vertexai
from vertexai.preview.language_models import TextGenerationModel
from google.cloud import storage
import csv

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Vertex AI Get API is running! Use /predict endpoint."

@app.route("/predict")
def predict():
    user_input = request.args.get("input")
    if not user_input:
        # (logic to read employees.csv from GCS here)
        user_input = "Summarize the employee dataset."

    vertexai.init(project="YOUR_PROJECT_ID", location="YOUR_REGION")
    model = TextGenerationModel.from_pretrained("text-bison")
    response = model.predict(user_input, max_output_tokens=256)
    return jsonify({"input": user_input, "output": response.text})
