import csv

file_name = input("Input file name: ")
newf_name = input("New file name: ")
input_file_path = f"/home/phanh/Downloads/finetuneGAMA/mistral7BData/{file_name}"
new_file_path = f"/home/phanh/Downloads/finetuneGAMA/mistral7BData/{newf_name}"
question = []
answer = []
new_answer = []

# Open the input file in read mode
with open(input_file_path, 'r') as f:
    data = csv.reader(f)
    header = next(data)  # Read the header
    for row in data:
        q, a = row
        question.append(q)
        answer.append(a)

    for a in answer:
        new_a = f"```gaml\n{a.strip()}\n```"
        new_answer.append(new_a)

new_data = [header, *zip(question, new_answer)]

# Open the new file in write mode and write the modified data
with open(new_file_path, "w", newline='') as nf:
    writer = csv.writer(nf)
    writer.writerows(new_data)
