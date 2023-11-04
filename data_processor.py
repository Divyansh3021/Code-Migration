import csv

# Define the CSV file names
input_csv_file = "Data/code_to_code_geekforgeek.csv"
output_csv_file = 'Data/processed_dataset.csv'

# Initialize lists to store the extracted data
input_code_list = []
input_language_list = []
output_code_list = []
output_language_list = []

# Read the CSV file and extract data
with open(input_csv_file, newline='') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        input_text = row['input_text']
        output_text = row['output_text']

        if "Translate Java to C++" in input_text:
            input_language = "Java"
            output_language = "C++"
        elif "Translate C++ to Java" in input_text:
            input_language = "C++"
            output_language = "Java"
        elif "Translate Python to C++" in input_text:
            input_language = "Python"
            output_language = "C++"
        elif "Translate Python to Java" in input_text:
            input_language = "Python"
            output_language = "Java"
        elif "Translate Java to Python" in input_text:
            input_language = "Java"
            output_language = "Python"
        elif "Translate C++ to Python" in input_text:
            input_language = "C++"
            output_language = "Python"
        else:
            # Handle other translation cases as needed
            continue

        # Append data to respective lists
        input_code_list.append(input_text)
        input_language_list.append(input_language)
        output_code_list.append(output_text)
        output_language_list.append(output_language)

# Save the processed data to a new CSV file
with open(output_csv_file, 'w', newline='') as csvfile:
    fieldnames = ['input_code', 'input_language', 'output_code', 'output_language']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(input_code_list)):
        writer.writerow({
            'input_code': input_code_list[i],
            'input_language': input_language_list[i],
            'output_code': output_code_list[i],
            'output_language': output_language_list[i]
        })

print(f"Processed data saved to {output_csv_file}")
