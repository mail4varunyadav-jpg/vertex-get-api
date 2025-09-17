import os
import csv
from flask import Flask, request, jsonify
from google.cloud import storage
import vertexai
from vertexai.language_models import TextGenerationModel

app = Flask(__name__)

PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION", "us-central1")
BUCKET = os.environ.get("BUCKET")

# Initialize Vertex AI SDK
vertexai.init(project=PROJECT_ID, location=REGION)
model = TextGenerationModel.from_pretrained("text-bison@002")

@app.route("/predict", methods=["GET"])
def predict():
    user_input = request.args.get("input")

    if not user_input:
        # If no input provided, load employees.csv from GCS
        client = storage.Client()
        bucket = client.bucket(BUCKET)
        blob = bucket.blob("employees.csv")

        if not blob.exists():
            return jsonify({"error": "employees.csv not found in bucket"}), 404

        data = blob.download_as_text().splitlines()
        reader = csv.DictReader(data)

        # Convert to list of dicts for summarization
        employees = [row for row in reader]

        # Build a prompt for Vertex AI
        user_input = f"""
        Here is an employee dataset with {len(employees)} records:
        {employees}

        Summarize the employee distribution by department and role.
        Provide insights like which departments have higher salaries.
        """

    # Call Vertex AI text-bison
    response = model.predict(prompt=user_input, temperature=0.2, max_output_tokens=300)

    return jsonify({
        "input": user_input,
        "output": response.text
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
