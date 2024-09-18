import pandas as pd
input_file = 'assignment 3.xlsx'
df =pd.read_excel(input_file)
output_file = 'assignment 3.csv'
df.to_csv(output_file, index=False)
print("conversion completed successfully.")
