import pandas as pd
import json
import numpy as np
from fuzzywuzzy import process, fuzz
from math import radians, sin, cos, sqrt, atan2

# File paths (adjust if needed)
INVENTORY_FILE = "realistic_inventory_with_service_time.csv"
SHOPDATA_FILE = "shopdata_with_time.csv"
FUZZY_TRAINING_FILE = "fuzzy_training_data.json"

# Load datasets
inventory_df = pd.read_csv(INVENTORY_FILE)
shopdata_df = pd.read_csv(SHOPDATA_FILE)

# Extract unique products and categories for fuzzy matching
unique_products = inventory_df["productName"].unique().tolist()
unique_categories = shopdata_df["category"].unique().tolist()

# Load or initialize fuzzy training data
try:
    with open(FUZZY_TRAINING_FILE, "r") as file:
        fuzzy_training_data = json.load(file)
except (FileNotFoundError, json.JSONDecodeError):
    fuzzy_training_data = {"categories": {}, "products": {}}

def train_fuzzy_match(query, actual, category_type):
    if query not in fuzzy_training_data[category_type]:
        fuzzy_training_data[category_type][query] = actual
    with open(FUZZY_TRAINING_FILE, "w") as file:
        json.dump(fuzzy_training_data, file, indent=4)

def fuzzy_match(query, choices, category_type, threshold=75):
    if query in fuzzy_training_data[category_type]:
        return fuzzy_training_data[category_type][query]
    match, score = process.extractOne(query, choices, scorer=fuzz.token_sort_ratio)
    if score >= threshold:
        train_fuzzy_match(query, match, category_type)
        return match
    return None

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth's radius in meters
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    a = max(0, min(1, a))  # Clamp a between 0 and 1
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def convert_numpy_types(data):
    """
    Recursively convert numpy data types (like np.int64) to native Python types.
    """
    if isinstance(data, dict):
        return {k: convert_numpy_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    elif isinstance(data, np.generic):
        return data.item()
    else:
        return data

def evaluate_recommendations(test_input: dict, filter_choice: int, selection_type: str, 
                             user_location: tuple, selected_path_index: int = 0):
    """
    Evaluate shop recommendations based on input data.

    Parameters:
      - test_input: dict mapping category names to items, e.g. {"meat": "lobster", "grooming": "mens harcut"}
      - filter_choice: int (1 for time-based filtering on queue_size, 2 for price-based, else rating-based)
      - selection_type: string indicating evaluation type ("categorical" or "manual")
      - user_location: tuple (lat, lon) provided by the frontend
      - selected_path_index: index (0-based) to select one of the generated shop paths (default=0)

    Returns:
      A dictionary with:
        - "selected_path": the chosen shop path (with updated token numbers),
        - "possible_paths": list of all generated paths,
        - "evaluationType": the provided selection type.
    """
    matched_results = {}
    # Fuzzy match categories and products
    for category, product in test_input.items():
        matched_category = fuzzy_match(category, unique_categories, "categories")
        matched_product = fuzzy_match(product, unique_products, "products")
        if not matched_product:
            # If no match found for product, try to match among products in the matched category if possible.
            if "category" in inventory_df.columns and matched_category:
                category_products = inventory_df[inventory_df["category"] == matched_category]["productName"].unique()
                if len(category_products) > 0:
                    matched_product = process.extractOne(product, category_products, scorer=fuzz.token_sort_ratio)[0]
        if matched_category and matched_product:
            matched_results[matched_category] = matched_product

    # Find relevant shops for each matched product
    final_shop_recommendations = {}
    for category, product in matched_results.items():
        product_shops = inventory_df[inventory_df["productName"] == product].merge(
            shopdata_df, on="shopId", how="inner"
        )
        if not product_shops.empty:
            product_shops["distance"] = product_shops.apply(
                lambda row: haversine_distance(user_location[0], user_location[1],
                                               row["latitude"], row["longitude"]), axis=1
            )
            # Sort by rating descending and take top 10 shops
            final_shop_recommendations[product] = product_shops.sort_values(by="rating", ascending=False).head(10)

    # Apply filtering based on user selection:
    for product, shops in final_shop_recommendations.items():
        if filter_choice == 1:
            final_shop_recommendations[product] = shops.sort_values(by="queue_size").head(10)
        elif filter_choice == 2:
            final_shop_recommendations[product] = shops.sort_values(by="price").head(10)
        else:
            final_shop_recommendations[product] = shops.sort_values(by="rating", ascending=False).head(10)

    # Generate 5 distinct optimal paths
    possible_paths = []
    for _ in range(5):
        path_shops = []
        chosen_shops = set()
        for product, shops in final_shop_recommendations.items():
            available_shops = shops[~shops["shopId"].isin(list(chosen_shops))]
            available_shops = available_shops.sort_values(by="rating", ascending=False)
            if not available_shops.empty:
                selected_shop = available_shops.iloc[0]
                chosen_shops.add(selected_shop["shopId"])
                # Use the actual coordinates from the merged DataFrame
                path_shops.append({
                    "shopId": selected_shop["shopId"],
                    "product": product,
                    "store": selected_shop["store"],
                    "rating": selected_shop["rating"],
                    "price": selected_shop["price"],
                    "distance": selected_shop["distance"],
                    "queue_size": selected_shop["queue_size"],
                    "lat": selected_shop["latitude"],
                    "long": selected_shop["longitude"],
                })
        possible_paths.append(path_shops)

    if not possible_paths or not possible_paths[selected_path_index]:
        result = {"message": "No valid shop path found.", "possible_paths": possible_paths}
        return convert_numpy_types(result)

    # Select the desired path and update inventory and queue sizes
    selected_path = possible_paths[selected_path_index]
    for shop in selected_path:
        shop_id = shop["shopId"]
        product = shop["product"]
        inventory_df.loc[
            (inventory_df["shopId"] == shop_id) & (inventory_df["productName"] == product),
            "stockAvailability"
        ] -= 1
        inventory_df["stockAvailability"] = inventory_df["stockAvailability"].clip(lower=0)
        shopdata_df.loc[shopdata_df["shopId"] == shop_id, "queue_size"] += 1
        new_token_number = shopdata_df.loc[shopdata_df["shopId"] == shop_id, "queue_size"].values[0]
        shop["new_token_number"] = new_token_number

    # Save updates back to CSV files
    inventory_df.to_csv(INVENTORY_FILE, index=False)
    shopdata_df.to_csv(SHOPDATA_FILE, index=False)

    result = {
        "selected_path": selected_path,
        "possible_paths": possible_paths,
        "evaluationType": selection_type,
    }
    return convert_numpy_types(result)

if __name__ == "__main__":
    test_input = {"meat": "lobster", "grooming": "mens harcut"}
    result = evaluate_recommendations(test_input, 1, "categorical", user_location=(20.3488, 85.8162))
    print(json.dumps(result, indent=2))
