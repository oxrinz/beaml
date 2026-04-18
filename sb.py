import sys
sys.path.insert(0, "tinygrad")

from tinygrad import Tensor, Device
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.dtype import dtypes
from tinygrad.renderer.amd.dsl import s, v, NULL
from tinygrad.runtime.autogen.amd.rdna4.ins import *

N = 64

def fill_kernel(Out):
  lidx = UOp.special(N, "lidx0")
  sink = UOp.sink(Out.base, lidx, arg=KernelInfo("fill"))
  insts = [
    s_load_b64(sdata=s[2:3], sbase=s[0:1], ioffset=0, soffset=NULL),  # load output ptr from kernarg
    s_wait_kmcnt(simm16=0),                                             # wait for scalar load
    v_mov_b32_e32(v[1], 0x3f800000),                                    # v1 = 1.0f
    v_lshlrev_b32_e32(v[2], 2, v[0]),                                   # v2 = tid * 4 (byte offset)
    v_mov_b32_e32(v[3], 0),                                             # v3 = 0 (upper 32 bits)
    global_store_b32(vaddr=v[2:3], vsrc=v[1], saddr=s[2:3]),           # out[tid] = 1.0f
    s_wait_storecnt(simm16=0),
    s_endpgm(),
  ]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT),
             UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=x) for x in insts))))

out = Tensor.empty(N, dtype=dtypes.float32)
out.realize()
out = Tensor.custom_kernel(out, fxn=fill_kernel)[0]
result = out.numpy()
print(result)
assert (result == 1.0).all(), f"FAIL: {result}"
print("OK!")
