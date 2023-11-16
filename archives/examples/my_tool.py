import Rappture
import sys

rx = Rappture.PyXml(sys.argv[1])
print rx['input.number(Ef).current'].value
rx.close()