import os
import polars as pl
import time
# Assuming tx_files is defined somewhere above this code
# Also assuming col is a list of columns and stations_list_TX is defined
tx_files = os.listdir('D:/Github_extras/Texas Weather/')
all_data = pl.DataFrame()
# col = ['UTCISO8601','WhoAmI','DewPointF','tempF','GlobalHorizontalIrradianceWM2','CloudCoverPerc','DirectHorizontalIrradianceWM2','WindSpeedmph','WindDirection','WindSpeed100mph']

# # start = time.time()
# try:
#     for file in tx_files:
#         if file.endswith('.csv'):
#             print(file)
#             if any(file[:-4] in x for x in tx_files) and (int(file[:4]) > 1999) and (int(file[:4]) < 2024):
#                 filepath = os.path.join('D:/Github_extras/Texas Weather/', f'{file}')
#                 print(filepath)
#                 data = pl.read_csv(filepath, has_header=True)
#                 # print(data.head)
#                 # delete first column and rename columns
#                 # data = data.drop(['Unnamed: 0'])
#                 # data.columns = col
#
#                 # # convert 'WhoAmI' column to string type
#                 # data = data.with_column(pl.col('WhoAmI').cast(pl.Utf8))
#                 one_file_read = time.time()
#                 # print(f"Time taken to read one file: {one_file_read-start}")
#                 all_data = pl.concat([all_data, data], rechunk=True)
#     # all_data.write_csv('D:/Github_extras/All_Weather_TX_1940_1980.csv')
#     all_data.write_parquet('D:/Github_extras/All_Weather_TX_2000_2023.parquet')
# except Exception as e:
#     print(e)
#     # all_data.write_csv('D:/Github_extras/All_Weather_TX_error_3.csv')
#     all_data.write_parquet('D:/Github_extras/All_Weather_TX_errors_3.parquet')



# comb_data = pl.read_parquet('D:/Github_extras/All_Weather_TX_errors_2.parquet')
# fin_data = pl.concat([comb_data, all_data], rechunk=True)
# fin_data.write_csv('D:/Github_extras/All_Weather_TX_combined_3.csv')