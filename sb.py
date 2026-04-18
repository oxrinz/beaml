import sys
sys.path.insert(0, "tinygrad")

from dataclasses import replace
from tinygrad import Tensor, Device
from tinygrad.uop.ops import Ops, pyrender
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.codegen import get_program
from tinygrad.engine.realize import CompiledRunner

ren = Device[Device.DEFAULT].renderer

# --- kernel AST ---
a = Tensor.rand(512, 512).realize()
b = Tensor.rand(512, 512).realize()
ast = (a @ b).schedule()[-1].ast

print(ast)
