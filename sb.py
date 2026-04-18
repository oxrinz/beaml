import sys
sys.path.insert(0, "tinygrad")

from dataclasses import replace
import anthropic
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
if ast.op is Ops.BEAM: ast = ast.src[0]

# --- base ProgramSpec (unoptimized) for metadata ---
s = Scheduler(ast, ren)
s.convert_loop_to_global()
prog = get_program(s.get_optimized_ast(name_override="matmul"), ren)

print(f"function_name: {prog.function_name}")
print(f"global_size:   {prog.global_size}")
print(f"local_size:    {prog.local_size}")
print(f"\nbase src:\n{prog.src}\n")

# --- ask the LLM to generate an optimized version ---
client = anthropic.Anthropic()

msg = client.messages.create(
  model="claude-sonnet-4-6",
  max_tokens=4096,
  messages=[{
    "role": "user",
    "content": f"""Here is an unoptimized Metal GPU kernel for a 512x512 matrix multiplication:

{prog.src}

The kernel function must be named exactly `{prog.function_name}`.
The buffer arguments must stay exactly the same.

Write an optimized Metal kernel for this. Use threadgroups, simdgroup matrix multiply (simdgroup_multiply_accumulate), vectorized loads (float4), shared memory, or whatever helps most. Return ONLY the kernel source code, no explanation."""
  }]
)

llm_src = msg.content[0].text.strip()
if llm_src.startswith("```"): llm_src = llm_src.split("\n", 1)[1].rsplit("```", 1)[0].strip()
print(f"=== LLM kernel ===\n{llm_src}\n")
print(f"=== first 3 lines ===")
for i, line in enumerate(llm_src.splitlines()[:3]): print(f"{i}: {repr(line)}")

# --- swap in LLM source and compile ---
llm_prog = replace(prog, src=llm_src, lib=None)
runner = CompiledRunner(llm_prog)
print(f"compiled: {len(runner.p.lib)} bytes")
