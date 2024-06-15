# Imported Libraries

import struct
import numpy as np  # For expensive data calculations
import xarray  # (opens data ) => Reads the NC files and converts them to a dataframe

# Imported Functions
from pandas import to_datetime, read_csv

# Imported Constants
from pytz import UTC

# Imported Types
from pandas import DataFrame
from datetime import datetime
from typing import Callable

# Custom Functions
# from Timing import timeit


def read_data(file_name: str) -> DataFrame:
    return xarray.open_dataset(file_name).to_dataframe()


def to_str(x, lens) -> str:
    if (x // 100) == 0:
        return f"{x:.2f}".zfill(lens)
    elif ((-x) // 100) == 0:
        return f"{x:.2f}".zfill(lens + 1)
    else:
        return f"{x:.2f}"



def clean_data(df: DataFrame) -> DataFrame:
    '''
        Pandas operations that prepare the data for insertion into a POWERWORLD file
    '''
    # flat the dataset for pd
    df['sped'] = np.sqrt(df['u10'] ** 2 + df['v10'] ** 2)
    df['sped100'] = np.sqrt(df['u100'] ** 2 + df['v100'] ** 2)
    df['drct'] = np.arctan2(df['u10'], df['v10'])
    df['drct'].where(df['drct'].isnull(), np.arctan2(df['u10'], df['v10']))
    df['tcc'].where(df['tcc'].isnull(), (df['hcc'] + df['mcc'] + df['lcc']) / 3)

    # ......... UNIT CONVERSION
    df['t2m'] = np.round((df['t2m'] - 273.15) * 9 / 5 + 32)  # convert to degF
    df['d2m'] = np.round((df['d2m'] - 273.15) * 9 / 5 + 32)  # convert to degF
    df['sped'] = np.round(df['sped'] * 2.23694)  # convert from mps to mph
    df['tcc'] = np.round(df['tcc'] * 100)  # convert to %
    df['drct'] = np.round(df['drct'] * 180 / np.pi + 180)  # convert to deg
    df['ssrd'] = df['ssrd'] / 3600  # J/m^2 => W/m^2
    df['fdir'] = df['fdir'] / 3600  # J/m^2 => W/m^2
    df['WindSpeed100mph'] = df['sped100'] * 2.23694  # convert from mps to mph
    if 'z' in df.columns: df['z'] = df['z'] / 9.81  # Something to do with Gravity?
    # ......... UNIT CONVERSION

    df = df.reset_index()
    df['station_id'] = '+' + df['latitude'].apply(to_str, args=(5,)) + df['longitude'].apply(to_str, args=(
    6,)) + '/'  # formate to PW auxiliary file
    df = df.drop(columns=[
        'longitude', 'latitude', 'u100', 'sped100', 'v100', 'u10', 'v10', 'hcc', 'mcc', 'lcc', 'z'
    ], errors='ignore')  # If 'ignore', suppress error and only existing labels are dropped.

    # ......... CONVERT TO PW PARAMETERS
    df['time'] = np.datetime_as_string(df['time'], unit='ms', timezone='UTC')
    df = df.rename(columns={
        "time": "UTCISO8601",
        "station_id": "WhoAmI",
        't2m': 'tempF',
        'd2m': 'DewPointF',
        "sped": "WindSpeedmph",
        "drct": "WindDirection",
        'tcc': 'CloudCoverPerc',
        'ssrd': 'GlobalHorizontalIrradianceWM2',
        'fdir': 'DirectHorizontalIrradianceWM2'
    })
    df = df.reindex(columns=[
        'UTCISO8601', 'WhoAmI', 'DewPointF', 'tempF', 'GlobalHorizontalIrradianceWM2', 'CloudCoverPerc',
        'DirectHorizontalIrradianceWM2', 'WindSpeedmph', 'WindDirection', 'WindSpeed100mph'
    ])
    df = df.sort_values(by=['UTCISO8601'])
    df['UTCISO8601'] = np.where(df['UTCISO8601'].duplicated(), '', df['UTCISO8601'])
    df['GlobalHorizontalIrradianceWM2'] = df[
        'GlobalHorizontalIrradianceWM2'].ffill()  # Fixes Global column for nans and spaces
    df['DirectHorizontalIrradianceWM2'] = df[
        'DirectHorizontalIrradianceWM2'].ffill()  # Fixed Direct column for nans and spaces
    df = df.fillna('')
    df = df.astype({
        'WhoAmI': 'str',
        'tempF': 'int16',
        'DewPointF': 'int16',
        'WindSpeedmph': 'int16',
        'WindDirection': 'int16',
        'CloudCoverPerc': 'int16',
        'GlobalHorizontalIrradianceWM2': 'int16',
        'DirectHorizontalIrradianceWM2': 'int16',
        'WindSpeed100mph': 'int16'
    })
    # ......... CONVERT TO PW PARAMETERS
    return df



def write_data(df: DataFrame, file_name: str) -> None:
    def datetime_to_excel_double(date_obj):
        excel_epoch = datetime(1899, 12, 31, tzinfo=UTC)  # Make it timezone-aware by setting the timezone to UTC
        delta = date_obj - excel_epoch
        excel_date = delta.days + (delta.seconds / 86400.0) + (date_obj.microsecond / 86400e6) + 1
        return excel_date

    # STATION OPERATIONS ...
    ### Loads in station.csv ###
    df_station: DataFrame = read_csv("station.csv")
    df_station['Region'] = df_station['Region'].fillna('')
    df_station['Country2'] = df_station['Country2'].fillna('')
    df_station['ElevationMeters'] = df_station['ElevationMeters'].astype(int)
    df_station['Region'] = df_station['Region'].astype(str)
    df_station['Country2'] = df_station['Country2'].astype(str)
    df_station.reset_index(inplace=True)
    df_station['WhoAmI'] = '+' + df_station['Latitude'].apply(to_str, args=(5,)) + df_station['Longitude'].apply(to_str,
                                                                                                                 args=(
                                                                                                                 6,)) + '/'
    df_station['WhoAmI'].drop_duplicates(inplace=True)
    df_station = df_station.sort_values(by=['WhoAmI'])
    LOC: int = df_station['WhoAmI'].nunique()

    to_cstring: Callable[[str], str] = lambda s: s.encode('ascii', 'replace') + b'\x00'
    df_station['ascii_null_terminated_WhoAmI'] = df_station['WhoAmI'].apply(to_cstring)
    df_station['ascii_null_terminated_Region'] = df_station['Region'].apply(to_cstring)
    df_station['ascii_null_terminated_Country2'] = df_station['Country2'].apply(to_cstring)
    # ... STATION OPERATIONS

    # FINAL DF OPERATIONS ...
    df['UTCISO8601'] = df['UTCISO8601'].ffill()
    df = df.ffill()
    df['UTCISO8601'] = to_datetime(df['UTCISO8601'], format='ISO8601').dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    df['UTCISO8601'] = to_datetime(df['UTCISO8601'])
    df['excel_double_datetime'] = df['UTCISO8601'].apply(datetime_to_excel_double)
    df = df.sort_values(by=['excel_double_datetime', 'WhoAmI'])
    COUNT = df['excel_double_datetime'].nunique()
    UNIQUE_DATES = df['excel_double_datetime'].unique()
    df['tempF102'] = df['tempF'] #+ 115
    df['DewPointF104'] = df['DewPointF'] + 115
    df['WindDirection107'] = df['WindDirection'] / 5
    df['GlobalHorizontalIrradianceWM2_120'] = df['GlobalHorizontalIrradianceWM2'] / 5
    df['DirectHorizontalIrradianceWM2_121'] = df['DirectHorizontalIrradianceWM2'] / 5
    df['WindDirection107'] = df['WindDirection107'].astype(int)
    df['GlobalHorizontalIrradianceWM2_120'] = df['GlobalHorizontalIrradianceWM2_120'].astype(int)
    df['DirectHorizontalIrradianceWM2_121'] = df['DirectHorizontalIrradianceWM2_121'].astype(int)
    df['tempF102'] = df['tempF102'].astype(int)  # just for 197605
    df['WindSpeedmph'] = df['WindSpeedmph'].astype(int)  # just for 197605
    df['CloudCoverPerc'] = df['CloudCoverPerc'].astype(int)  # just for 197605
    df['WindSpeed100mph'] = df['WindSpeed100mph'].astype(int)  # just for 197605
    # ... FINAL DF OPERATIONS

    # WRITING TO THE PWW FILE ...
    aPWWVersion = 1

    # Extracting the smallest start time and the largest end time from the DataFrame and converting them to timestamps
    aStartDateTimeUTC = df['excel_double_datetime'].min()
    aEndDateTimeUTC = df['excel_double_datetime'].max()
    # area=[58, -130, 24, -60] North 58°, West -130°, South 24°, East -60°
    aMinLat = int(df_station['Latitude'].min())
    aMaxLat = int(df_station['Latitude'].max())
    aMinLon = int(df_station['Longitude'].min())
    aMaxLon = int(df_station['Longitude'].max())
    # Define the optional identifier field count

    print("Latitude: ", aMinLat, aMaxLat)
    print("Longitude: ", aMinLon, aMaxLon)

    LOC_FC: int = 0  # for extra loc variables from table 1
    VARCOUNT: int = 8  # Set this to the number of weather variable types you have

    with open(file_name, 'wb') as file:
        # ......... VOODOO MAGIC
        file.write(struct.pack('<h', 2001))
        file.write(struct.pack('<h', 8065))
        file.write(struct.pack('<h', aPWWVersion))
        file.write(struct.pack('<d', aStartDateTimeUTC))
        file.write(struct.pack('<d', aEndDateTimeUTC))
        file.write(struct.pack('<d', aMinLat))
        file.write(struct.pack('<d', aMaxLat))
        file.write(struct.pack('<d', aMinLon))
        file.write(struct.pack('<d', aMaxLon))
        file.write(struct.pack('<h', 0))
        file.write(struct.pack('<i', COUNT))  # countNumber of datetime values (COUNT)
        file.write(struct.pack('<i', 3600))
        file.write(struct.pack('<i', LOC))  # Number of weather measurement locations (LOC)
        file.write(struct.pack('<h', LOC_FC))  # Loc_FC # Pack the data into INT16 format and write to stream
        file.write(struct.pack('<h', VARCOUNT))
        file.write(struct.pack('<h', 102))  # Temp in F
        file.write(struct.pack('<h', 104))  # Dew point in F
        file.write(struct.pack('<h', 106))  # Wind speed at surface (10m) in mph
        file.write(struct.pack('<h', 107))  # Wind direction at surface (10m) in 5-degree increments
        file.write(struct.pack('<h', 119))  # Total cloud cover percentage
        file.write(struct.pack('<h', 110))  # Wind speed at 100m in mph
        file.write(struct.pack('<h', 120))  # Global Horizontal Irradiance in W/m^2 divided by 4
        file.write(struct.pack('<h', 121))  # Direct Horizontal Irradiance in W/m^2 divided by 4
        file.write(struct.pack('<h', 8))  # BYTECOUNT
        # ......... VOODOO MAGIC

        # for date in UNIQUE_DATES:
        #     file.write(struct.pack('<d', date))

        for row in df_station.index:
            file.write(struct.pack('<d', df_station['Latitude'][row]))  # Write Latitude (DOUBLE)
            file.write(struct.pack('<d', df_station['Longitude'][row]))  # Write Longitude (DOUBLE)
            file.write(struct.pack('<h', df_station['ElevationMeters'][row]))  # Write AltitudeM (INT16)
            file.write(df_station['ascii_null_terminated_WhoAmI'][row])  # Write Name (CSTRING)
            file.write(df_station['ascii_null_terminated_Country2'][row])
            file.write(df_station['ascii_null_terminated_Region'][row])

        for date in UNIQUE_DATES:
            # Filter rows by unique date
            rows: DataFrame = df[df['excel_double_datetime'] == date]

            for temp in rows['tempF102']:
                file.write(int(temp).to_bytes(1, 'little'))

            for dew_point in rows['DewPointF104']:
                file.write(int(dew_point).to_bytes(1, 'little'))

            for wind_speed in rows['WindSpeedmph']:
                file.write(int(wind_speed).to_bytes(1, 'little'))

            for wind_direction in rows['WindDirection107']:
                file.write(int(wind_direction).to_bytes(1, 'little'))

            for CloudCoverPerc in rows['CloudCoverPerc']:
                file.write(int(CloudCoverPerc).to_bytes(1, 'little'))

            for WindSpeed100mph in rows['WindSpeed100mph']:
                file.write(int(WindSpeed100mph).to_bytes(1, 'little'))

            for GlobalHorizontalIrradianceWM2_120 in rows['GlobalHorizontalIrradianceWM2_120']:
                file.write(int(GlobalHorizontalIrradianceWM2_120).to_bytes(1, 'little'))

            for DirectHorizontalIrradianceWM2_121 in rows['DirectHorizontalIrradianceWM2_121']:
                file.write(int(DirectHorizontalIrradianceWM2_121).to_bytes(1, 'little'))
    # WRITING TO THE PWW FILE ...


# Read the data to a dataframe
df = read_data("Texas2024_Q2.nc")

# Process the data and write to a CSV
clean_data(df).to_csv("Texas_Q2.csv")

# Read the data from the CSV and remove the file
df2 = read_csv("Texas_Q2.csv")

# Process the data and write it to a usable PWW format
write_data(df2, "Texas_Q2_4.pww")