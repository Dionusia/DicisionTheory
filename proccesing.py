import pandas as pd

# Load the Excel file
file_path = 'BreastTissue.xlsx'
excel_data = pd.read_excel(file_path, sheet_name='Data')

# Identify numerical columns
numerical_columns = excel_data.select_dtypes(include=['float64', 'int64']).columns

# Normalize only the numerical columns to the range [-1, 1]
excel_data[numerical_columns] = 2 * (excel_data[numerical_columns] - excel_data[numerical_columns].min()) / (
        excel_data[numerical_columns].max() - excel_data[numerical_columns].min()) - 1

# Create a mapping for the "Class" column
class_mapping = {
    'car': '1',
    'fad': '2',
    'mas': '3',
    'gla': '4',
    'con': '5',
    'adi': '6'
}

# Replace the "Class" column values with the mapping values
excel_data['Class'] = excel_data['Class'].map(class_mapping)

# Save the normalized data to a CSV file
csv_file_path = 'BreastTissue.csv'
excel_data.to_csv(csv_file_path, index=False)

# Display the first few rows of the normalized data
print(excel_data.head())