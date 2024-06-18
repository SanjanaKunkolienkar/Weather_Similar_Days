from esa import saw
import os
import logging
import win32com.client
import pandas as pd

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

try:
    logger.info("Trying to connect to PowerWorld")
    pw_object = win32com.client.Dispatch("pwrworld.SimulatorAuto")
except Exception as e:
    logger.error(f"Error connecting to PowerWorld: {e}")

result = pw_object.RunScriptCommand('NewCase;')
logger.info(f"{CheckResultForError(result, 'New Case created')}")

result = pw_object.ProcessAuxFile(aux_filepath)
logger.info(f"{CheckResultForError(result, 'Aux files loaded successfully')}")

result = pw_object.RunScriptCommand('SolvePowerFlow(DC);')
logger.info(f"{CheckResultForError(result, 'Solved Power Flow')}")

# get a list of files in the pww_filepath directory
files = os.listdir(pww_filepath)

var_list = ['Date', 'Hour', 'TimeDomainWeatherSummary', 'TimeDomainWeatherSummary:1', 'TimeDomainWeatherSummary:2']
for i in range(1, 2680):
    var_list.append(f'TimeDomainWeatherTemp:{i}')
i=0
df_results = pd.DataFrame(columns=var_list)
print(df_results.head())
#pd.DataFrame(columns = ['Date', 'TempAvg', 'TempMin', 'TempMax', 'DewPointAvg', 'DewPointMin', 'DewPointMax',
              #    'WindSpeedAvg', 'WindSpeedMin', 'WindSpeedMax', 'WindDirAvg', 'WindDirMin', 'WindDirMax',
               #   'CloudCoverAvg', 'CloudCoverMin', 'CloudCoverMax', 'WindSpeed100Avg', 'WindSpeed100Min', 'WindSpeed100Max',
                #  'GlobHorIrradAvg', 'GlobHorIrradMin', 'GlobHorIrradMax', 'DirNormIrradAvg', 'DirNormIrradMin', 'DirNormIrradMax'])
for file in files:
    print("Filename: ", file[:-4])
    pww_file = os.path.join(pww_filepath, file)
    command = 'TimeStepAppendPWW("{}", Single Solution)'.format(pww_file)
    result_timestep = pw_object.RunScriptCommand(command)
    logger.info(f"{CheckResultForError(result_timestep, 'PWW file loaded successfully')}")

    result_temp = pw_object.GetParametersMultipleElement('TempF', var_list, '')
    logger.info(f"{CheckResultForError(result_temp, 'Data saved successfully')}")
    data = result_temp[1]
    print(data)
    df_data = pd.DataFrame({f"Column{i + 1}": [x.strip() if x is not None else None for x in column] for i, column in
                              enumerate(data)})
    #rename columns
    df_data.columns = var_list
    print(df_data.head(5))
    #conver all columns except Date to float
    df_data.iloc[:, 1:] = df_data.iloc[:, 1:].astype(float)

    command_del = 'TimeStepDeleteAll;'
    result_timestep = pw_object.RunScriptCommand(command_del)

    #df_byday = df_data.groupby('Date').mean()
    # combine dataframe df_byday for all for loops
    df_results = pd.concat([df_results, df_data])

df_results.to_csv('D:/Github_extras/Texas_1940-2023/Texas_1940_2023_test.csv')

