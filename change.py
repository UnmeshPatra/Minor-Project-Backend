import pandas as pd

# Load the Excel file
file_path = "shopdata_with_time.xlsx"  # Change this to your actual file path
df = pd.read_excel(file_path)

# Split the 'new_point' column into 'latitude' and 'longitude'
df[['latitude', 'longitude']] = df['new_point'].str.split(',', expand=True)

# Convert latitude and longitude back to float (optional but recommended)
df['latitude'] = df['latitude'].astype(float)
df['longitude'] = df['longitude'].astype(float)

# Save the updated dataframe
df.to_excel("updated_file.xlsx", index=False)

print("Latitude and Longitude columns updated successfully!")
