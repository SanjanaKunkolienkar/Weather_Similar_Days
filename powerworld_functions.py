import win32com.client
import os
import logging
import pythoncom

cwd = os.getcwd()
def connect_to_powerworld(logger, my_bar):
    try:
        pw_object = win32com.client.Dispatch("pwrworld.SimulatorAuto", pythoncom.CoInitialize())
        my_bar.progress(50, text='Connected to PowerWorld')
    except Exception as e:
        logger.info(f"Error connecting to PowerWorld: {e}")

    return pw_object

def CheckResultForError(SimAutoOutput, Message):
    # Test if powerworld sim auto works
    if SimAutoOutput[0] != '':
        print('Error: ' + SimAutoOutput[0])
    else:
        print(Message)
        return Message

def get_pww_files(pww_filepath, year, logger):
    files = os.listdir(pww_filepath)
    for file in files:
         # check if year is in file
        if str(year) in file:
            year_file = file

    logger.info(f"File names: {year_file}")
    return year_file

def get_data_from_pww(pw_object, pww_filepath, year_file, logger):

    result = pw_object.RunScriptCommand('NewCase;')
    logger.info(f"{CheckResultForError(result, 'New Case created')}")

    pww_file = os.path.join(pww_filepath, year_file, '.aux')
    command = 'TimeStepAppendPWW("{}", Single Solution)'.format(pww_file)
    result_timestep = pw_object.RunScriptCommand(command)
    logger.info(f"{CheckResultForError(result_timestep, 'PWW file loaded successfully')}")

    command, result_timestep = '', ''
    command = 'WeatherPWWSetDirectory("{}", YES)'.format(pww_filepath)
    result_timestep = pw_object.RunScriptCommand(command)
    logger.info(f"{CheckResultForError(result_timestep, 'PWW weather directory set')}")



def main(date, my_bar):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=os.path.join(cwd, 'progress1.log'), filemode='w', level=logging.INFO)
    logger = logging.getLogger()

    pw_object = connect_to_powerworld(logger)
    pww_filepath = os.path.join(os.getcwd(), 'Weather Aux By Years/Texas_ByTwoYears/')
    my_bar.progress(60, text='Retrieved file')
    # get year from date
    year = date[:4]
    files = get_pww_files(pww_filepath, year, logger)



if __name__ == '__main__':
    main(date, my_bar)