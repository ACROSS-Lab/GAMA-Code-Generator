import json
import csv

json_file = str(input("input json file: "))
csv_file = str(input("csv file: "))

# Read data from JSON file
with open(json_file, 'r') as file:
    data = json.load(file)


# Extract fieldnames (column names) from the data
fieldnames = list(data[0].keys()) if data else []

# Write data to CSV file
with open(csv_file, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    # Write header
    writer.writeheader()

    # Write rows
    for row in data:
        writer.writerow(row)

print("CSV file saved successfully.")

    
