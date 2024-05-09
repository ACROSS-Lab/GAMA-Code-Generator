import pandas as pd

if __name__ == '__main__':

    file_name = input("Input file name: ")
    newf_name = input("New file name: ")
    csv_file_path = f"/home/phanh/Downloads/finetuneGAMA/mistral7BData/{newf_name}"
    json_file_path= f"/home/phanh/Downloads/finetuneGAMA/mistral7BData/{file_name}"
    df = pd.read_json(json_file_path)
    df.to_csv(csv_file_path)

