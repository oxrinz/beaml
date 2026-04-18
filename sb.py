import sys, os
sys.path.insert(0, "tinygrad")
os.environ["AMD_AQL"] = "1"

from tinygrad import Device

dev = Device["AMD"]

print(dev)
