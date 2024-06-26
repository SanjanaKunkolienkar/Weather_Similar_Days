
import pythoncom
import win32com.client

import warnings
warnings.filterwarnings("ignore")
def CheckResultForError(SimAutoOutput, Message):
    # Test if powerworld sim auto works
    if SimAutoOutput[0] != '':
        print('Error: ' + SimAutoOutput[0])
    else:
        print(Message)
        return Message

try:
    pw_object = win32com.client.Dispatch("pwrworld.SimulatorAuto")
except Exception as e:
    print(f"Error connecting to PowerWorld: {e}")

result = pw_object.RunScriptCommand('NewCase;')
print(f"{CheckResultForError(result, 'New Case created')}")
command = 'TimeStepSaveResultsByTypeCSV(ObjectType, "FileCSVName");'
result = pw_object.RunScriptCommand(command)