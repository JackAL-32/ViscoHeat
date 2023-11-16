import Rappture
import sys

io = Rappture.PyXml(sys.argv[1])
print io['input.number(Amplitude).default'].value
io.close()
