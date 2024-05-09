import csv
import random
# Define the list of tuples
tuples_list = [('a', 'b'), ('c', 'd'), ('a', 'e'), ('b', 'f'), ('a', 'g'), ('c', 'h'), ('b', 'i'), ('a', 'j'), ('a', 'k'), ('c', 'l'), ('c', 'm'), ('a', 'n')]
new_list = [['a', 'x'], ['c', 'z'], ['b', 'w']]
# Create an empty dictionary to store values
grouped_values = {}

# Group values by 'a'
for a, b in tuples_list:
    if a in grouped_values:
        grouped_values[a].append(b)
    else:
        grouped_values[a] = [b]

    #print(grouped_values.get(a))


counts_by_first_element = {}


# Count values grouped by 'a', 'b', 'c', etc.
for a, b in tuples_list:
    if a in counts_by_first_element:
        counts_by_first_element[a] += 1
    else:
        counts_by_first_element[a] = 1

print("Counts by 'a':", counts_by_first_element)

values_to_choose = {}
for group, counts in counts_by_first_element.items():
    values_to_choose[group] = random.sample(grouped_values[group], min(2, counts))

# Print randomly chosen values from each group
for group, values in values_to_choose.items():
    print(f"Randomly chosen values from '{group}':", values)

# Combine the new and existing lists
combined_list = []
for item in new_list:
    if item[0] in values_to_choose:
        combined_list.append(item + values_to_choose[item[0]])
    else:
        combined_list.append(item + [''])

# Write data to CSV
with open("combined_data.csv", 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Group', 'Value 1', 'Value 2', 'Choose Value'])  # Write header

    # Write combined data to CSV
    for item in combined_list:
        writer.writerow(item)


# Write data to CSV
with open("randomly_chosen_values.csv", 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Group', 'Value 1', 'Value 2'])  # Write header

    # Write randomly chosen values to CSV
    for group, values in values_to_choose.items():
        writer.writerow([group] + values)

results = []
for a, b_values in grouped_values.items():
    if 'b' in b_values:
        results.append(('b', a))
    if 'e' in b_values:
        results.append(('e', a))

for a, b_values in grouped_values.items():
    print(f"Values for '{a}': {b_values}")

with open("combined_data.csv", 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Group', 'Value 1', 'Value 2', 'Choose Value'])  # Write header

    # Write combined data to CSV
    for item in combined_list:
        writer.writerow(item)

'''
with open("grouped_values.csv", 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['a', 'b_values'])  # Write header

    # Write each row with 'a' value and corresponding 'b' values
    for a, b_values in grouped_values.items():
        writer.writerow([a] + b_values)
# print(results)
'''