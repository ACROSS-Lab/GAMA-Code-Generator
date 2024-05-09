import csv
import json

def csv_to_json(csv_file_path, json_file_path):
	#create a dictionary
	data_array = []

	#Step 2
	#open a csv file handler
	with open(csv_file_path, encoding = 'utf-8') as csv_file_handler:
		csv_reader = csv.DictReader(csv_file_handler)

		#convert each row into a dictionary
		#and add the converted data to the data_variable

		for row in csv_reader:

			#assuming a column named 'No'
			#to be the primary key
			#key = rows['Serial Number']
			data_array.append(row)

	#open a json file handler and use json.dumps
	#method to dump the data
	#Step 3
			
	with open(json_file_path, 'w', encoding = 'utf-8') as json_file_handler:
		#Step 4
		json_file_handler.write(json.dumps(data_array, indent = 4))

#driver code
#be careful while providing the path of the csv file
#provide the file path relative to your machine

def merge_json_data()

#Step 1
file_name = input("Input file name: ")
newf_name = input("New file name: ")
csv_file_path = f"/home/phanh/Downloads/finetuneGAMA/data/train/csv/{file_name}"
json_file_path= f"/home/phanh/Downloads/finetuneGAMA/data/train/json/{newf_name}"


csv_to_json(csv_file_path, json_file_path)