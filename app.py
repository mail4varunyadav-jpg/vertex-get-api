from flask import Flask, request, jsonify
import os
from google.cloud import storage
import csv
import vertexai
from vertexai.language_models import TextGenerationModel

# -------------------
# Configuration
# -------------------
PROJECT_ID = "thinking-device-464711-n6"
REGION = "europe-west4"
BUCKET_NAME = "employee-dataset-bucket"  # replace with your GCS bucket
CSV_FILE = "employees.csv"        # replace with your CSV file name

# Make sure GOOGLE_APPLICATION_CREDENTIALS is set
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/service-account.json"
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID

# -------------------
# Initialize Vertex AI
# -------------------
vertexai.init(location=REGION)
model = TextGenerationModel.from_pretrained("gemini-2.0-flash")


# -------------------
# Flask app
# -------------------
app = Flask(__name__)

# Load CSV once on startup to avoid repeated GCS reads
def load_csv_from_gcs(bucket_name, file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    data = blob.download_as_text().splitlines()
    reader = csv.reader(data)
    return list(reader)

try:
    csv_data = load_csv_from_gcs(BUCKET_NAME, CSV_FILE)
    csv_text = "\n".join([", ".join(row) for row in csv_data])
except Exception as e:
    csv_text = ""
    print(f"⚠️ Failed to load CSV: {e}")

@app.route("/")
def home():
    return "✅ Vertex AI Get API is running! Use /predict endpoint."

@app.route("/predict")
def predict():
    user_input = request.args.get("input")
    
    # If no input, summarize the CSV
    if not user_input:
        if not csv_text:
            return jsonify({"error": "CSV data unavailable and no input provided."}), 400
        user_input = f"Summarize the following employee dataset:\n{csv_text}"

    # Call Vertex AI
    try:
        response = model.predict(user_input, max_output_tokens=256)
        output_text = response.text
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"input": user_input, "output": output_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)


