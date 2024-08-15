import os
import logging
import win32com.client
import pandas as pd
import pythoncom

cwd = os.getcwd()
pww_filepath = "D:/Github_extras/Texas_1940-2023/Texas_ByTwoYears/"
aux_filepath = "D:/Github_extras/Texas7k_20210804.AUX"

# Configure logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=os.path.join(cwd, 'progress.log'), filemode='w', level=logging.INFO)
logger = logging.getLogger()

def CheckResultForError(SimAutoOutput, Message):
    # Test if powerworld sim auto works
    if SimAutoOutput[0] != '':
        print('Error: ' + SimAutoOutput[0])
    else:
        print(Message)
        return Message

def get_generator_data():
    try:
        logger.info("Trying to connect to PowerWorld")
        pw_object = win32com.client.Dispatch("pwrworld.SimulatorAuto", pythoncom.CoInitialize())
    except Exception as e:
        logger.error(f"Error connecting to PowerWorld: {e}")

    result = pw_object.RunScriptCommand('NewCase;')
    logger.info(f"{CheckResultForError(result, 'New Case created')}")

    result = pw_object.ProcessAuxFile(aux_filepath)
    logger.info(f"{CheckResultForError(result, 'Aux files loaded successfully')}")

    result = pw_object.RunScriptCommand('SolvePowerFlow(DC);')
    logger.info(f"{CheckResultForError(result, 'Solved Power Flow')}")

    # get substation data from powerworld
    result = pw_object.GetParametersMultipleElement('Gen', ['BusNum', 'Latitude:1', 'Longitude:1', 'FuelType'], '')
    logger.info(f"{CheckResultForError(result, 'Substation data retrieved')}")

    gen_df = pd.DataFrame({f"Column{i + 1}": [x.strip() if x is not None else None for x in column] for i, column in
                              enumerate(result[1])})
    gen_df.columns = ['BusNum', 'latitude', 'longitude', 'FuelType']
    # print(gen_df.head)

    # print unique fuel types
    fuel_types = gen_df['FuelType'].unique()
    # print(fuel_types)

    ren = ['SUN (Solar)', 'WND (Wind)']
    # filter the dataframe for solar and wind generators
    gen_df = gen_df[gen_df['FuelType'].isin(ren)]
    # print(gen_df.head)

    gen_df.reset_index(drop=True, inplace=True)

    # convert latitude and longitude to float
    gen_df['latitude'] = gen_df['latitude'].astype(float)
    gen_df['longitude'] = gen_df['longitude'].astype(float)

    # Map numeric categories to specific colors
    color_map = {'SUN (Solar)': '#EDD83D', 'WND (Wind)': '#6AB547'}  # Hex color codes with a hash prefix
    gen_df['color'] = gen_df['FuelType'].map(color_map)  # Create a new column for colors

    return gen_df