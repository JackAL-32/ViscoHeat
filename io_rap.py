import Rappture
from sys import argv

io = ""

def start_io(argv):
    io = Rappture.PyXml(argv)

def get_io(group, number):
    return float(io[f'input.group(InputParameters).group({group}).number({number}).current'].value)

def put_io(internal_name: str, name: str, label: str):
    io[f'output.image({internal_name}).about.label']=label
    io.put(f'output.image({internal_name}).current', name + ".png",type='file',compress=True)

def close_io():
    if io != "":
        io.close()
