import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json()
    if not data:
        return jsonify({"message": "No data provided."}), 400

    option = data.get("option")
    selection_type = data.get("selectionType")  # "time" or "price"
    
    # Check which option was sent
    if option == "categorical":
        # Expecting "data" to be a list of dictionaries, e.g.:
        # [{ "category": "Beauty", "name": "Lipstick" }, ...]
        items = data.get("data")
        if not isinstance(items, list):
            return jsonify({"message": "Data should be a list of items."}), 400

        # Convert the list of items to a dictionary.
        # If duplicate categories are present, aggregate names in a list.
        parsed_data = {}
        for item in items:
            cat = item.get("category")
            name = item.get("name", "N/A")
            if cat in parsed_data:
                # If already a list, append; else convert to list
                if isinstance(parsed_data[cat], list):
                    parsed_data[cat].append(name)
                else:
                    parsed_data[cat] = [parsed_data[cat], name]
            else:
                parsed_data[cat] = name

        message = f"Categorical data parsed: {parsed_data}."
        return jsonify({"message": message, "evaluationType": selection_type})
    
    elif option == "manual":
        # For manual input, "data" is expected to be a string.
        manual_input = data.get("data")
        print(manual_input)
        if not isinstance(manual_input, str):
            return jsonify({"message": "Data should be a string for manual input."}), 400

        # Build the prompt using the manual input
        prompt = (
            f"You will be provided input which will contain one or more items. Convert the input into a dictionary of the where category is one of the following : Beauty, Clothing, Electronics, Groceries, Medicine, Meat. and then the item. The input will contain items and they need to be categorized in this format"
            f"{{Category1: Item1, Category2: Item2, ...}} Input: {manual_input}"
        )
        print(prompt)
        try:
            from google import genai
        except ImportError:
            return jsonify({"message": "Google GenAI library is not installed."}), 500

        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            return jsonify({"message": "Gemini API key is not set in the environment."}), 500

        # Initialize the Gemini client
        client = genai.Client(api_key=gemini_api_key)
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
        except Exception as e:
            return jsonify({"message": f"Error communicating with Gemini API: {e}"}), 500

        gemini_output = response.text
        return jsonify({"message": gemini_output, "evaluationType": selection_type})
    
    else:
        return jsonify({"message": "Invalid option provided."}), 400

if __name__ == "__main__":
    # Run the Flask app in debug mode.
    app.run(debug=True)