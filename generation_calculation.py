import pandas as pd
import os

cwd = os.getcwd()

path_gen_data = 'D:/Github_extras/Weather Aux By Years/AllYearsEIA2024Q1'

files = os.listdir(path_gen_data)

texas_df = pd.DataFrame()

for file in files:
    excel_file = os.path.join(path_gen_data, file)
    print(excel_file)

    df = pd.read_excel(excel_file, skiprows=1)

    # delete all columns except '48 Gen MW Wind' and '48 Gen MW Solar'
    df = df[['Date', 'Time', '48 Gen MW Wind', '48 Gen MW Solar']]

    texas_df = pd.concat([texas_df, df], axis=0)

texas_df = texas_df.reset_index(drop=True)

texas_df.to_csv('D:/Github_extras/Weather Aux By Years/Texas_EIA2024Q1.csv', index=False)


