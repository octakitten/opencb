import silky
from silky import iteration
from silky import train as tr

opts = tr.optionsobj("ILSVRC/imagenet-1k", None, "./testing01/", 256, 256, 200, 500, 1000, 2)
tr.time_chamber(opts)
