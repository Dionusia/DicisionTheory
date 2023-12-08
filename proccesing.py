import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#load xlsx file
file_path = 'BreastTissue.xlsx'
excel_data = pd.read_excel(file_path, sheet_name='Data')

#identify numerical columns
numerical_columns = excel_data.select_dtypes(include=['float64', 'int64']).columns

#normalize only the numerical columns to the range [-1, 1], except column Class
excel_data[numerical_columns] = 2 * (excel_data[numerical_columns] - excel_data[numerical_columns].min()) / (
        excel_data[numerical_columns].max() - excel_data[numerical_columns].min()) - 1

#check for NaN values
if excel_data.isna().any().any():
    print("The DataFrame contains NaN values. Please handle them before proceeding.")
else:
    print("No NaN values found in the DataFrame.")

#create a mapping for the "Class" column to do the match
class_mapping = {
    'car': '1',
    'fad': '2',
    'mas': '3',
    'gla': '4',
    'con': '5',
    'adi': '6'
}

#replacing the values
excel_data['Class'] = excel_data['Class'].map(class_mapping)

#save the data into a csv file
csv_file_path = 'BreastTissue.csv'
excel_data.to_csv(csv_file_path, index=False)

#display few rows to check
print(excel_data.head())

#create a heatmap using seaborn
corr_matrix = excel_data[numerical_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Dataset')
plt.show()

