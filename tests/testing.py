import silky
from silky import iteration
from silky import train as tr

opts = tr.optionsobj("ILSVRC/imagenet-1k", None, "", 256, 256, 500, 500, 200, 2)
tr.time_chamber(opts)
