import os
import re
import json
import ast
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from recommendation_module import evaluate_recommendations

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
    selection_type = data.get("selectionType")  # Expected to be "time" or "price"
    
    # Check for user location; expect an array with two numbers [lat, lon]
    user_location = data.get("user_location")
    if user_location and isinstance(user_location, list) and len(user_location) == 2:
        try:
            user_location = (float(user_location[0]), float(user_location[1]))
            print(user_location)
        except (ValueError, TypeError):
            return jsonify({"message": "Invalid user_location format. Must be [lat, lon]."}), 400
    else:
        # Fallback default location
        user_location = (20.3488, 85.8162)

    if option == "categorical":
        # Expecting "data" to be a list of dictionaries
        items = data.get("data")
        # If items is a string, try to convert it into a list using ast.literal_eval
        if isinstance(items, str):
            try:
                items = ast.literal_eval(items)
            except Exception as e:
                return jsonify({"message": f"Failed to parse data string as list: {e}"}), 400
        if not isinstance(items, list):
            return jsonify({"message": "Data should be a list of items."}), 400

        # Convert the list of items to a dictionary. If duplicate categories, keep the first occurrence.
        parsed_data = {}
        for item in items:
            cat = item.get("category")
            name = item.get("name", "N/A")
            if cat not in parsed_data:
                parsed_data[cat] = name

        filter_choice = 1 if selection_type == "time" else 2
        result = evaluate_recommendations(parsed_data, filter_choice, selection_type, user_location=user_location)
        return jsonify(result)
    
    elif option == "manual":
        # For manual input, "data" is expected to be a string.
        manual_input = data.get("data")
        if not isinstance(manual_input, str):
            return jsonify({"message": "Data should be a string for manual input."}), 400

        prompt = (
            f"You will be provided input which will contain one or more items. Convert the input into a dictionary "
            f"where the category is one of the following: Beauty, Clothing, Electronics, Groceries, Medicine, Meat "
            f"and then the item. The input will contain items and they need to be categorized in this format: "
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

        client = genai.Client(api_key=gemini_api_key)
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
        except Exception as e:
            return jsonify({"message": f"Error communicating with Gemini API: {e}"}), 500

        gemini_output = response.text
        print("Gemini output:", gemini_output)
        match = re.search(r"(\{.*?\})", gemini_output, re.DOTALL)
        if not match:
            return jsonify({"message": "No dictionary found in the Gemini output."}), 400
        try:
            parsed_data = json.loads(match.group())
        except json.JSONDecodeError:
            return jsonify({"message": "Error parsing the dictionary from Gemini output."}), 400
        
        filter_choice = 1 if selection_type == "time" else 2
        result = evaluate_recommendations(parsed_data, filter_choice, selection_type, user_location=user_location)
        return jsonify(result)
    
    else:
        return jsonify({"message": "Invalid option provided."}), 400

if __name__ == "__main__":
    app.run(debug=True)
