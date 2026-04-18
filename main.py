import sys
sys.path.insert(0, "tinygrad")

if __name__ == "__main__":
  from tinygrad import Tensor, Device
  from tinygrad.uop.ops import Ops
  from tinygrad.codegen.opt.postrange import Scheduler, bufs_from_ast
  from tinygrad.codegen.opt.search import beam_search
  from tinygrad.codegen import get_program

  ren = Device[Device.DEFAULT].renderer

  # --- build a matmul kernel AST ---
  a = Tensor.rand(512, 512)
  b = Tensor.rand(512, 512)
  ast = (a @ b).schedule()[-1].ast
  if ast.op is Ops.BEAM: ast = ast.src[0]

  rawbufs = bufs_from_ast(ast, Device.DEFAULT)

  # ------------------------------------------------------------------ #
  # BEFORE: unoptimized
  # ------------------------------------------------------------------ #
  s_base = Scheduler(ast, ren)
  prog_before = get_program(s_base.get_optimized_ast(name_override="before"), ren)

  print("=== BEFORE beam search ===")
  print(f"shape:        {s_base.colored_shape()}")
  print(f"full_shape:   {s_base.full_shape}")
  print(f"axis_types:   {[str(t).split('.')[1] for t in s_base.axis_types]}")
  print(f"applied_opts: {s_base.applied_opts}")
  print(f"global_size:  {prog_before.global_size}")
  print(f"local_size:   {prog_before.local_size}")
  print()
  print(prog_before.src)

  # ------------------------------------------------------------------ #
  # AFTER: beam search
  # ------------------------------------------------------------------ #
  s_base2 = Scheduler(ast, ren)
  result = beam_search(s_base2, rawbufs, amt=3)
  prog_after = get_program(result.get_optimized_ast(name_override="after"), ren)

  print("\n=== AFTER beam search ===")
  print(f"shape:        {result.colored_shape()}")
  print(f"full_shape:   {result.full_shape}")
  print(f"axis_types:   {[str(t).split('.')[1] for t in result.axis_types]}")
  print(f"applied_opts: {result.applied_opts}")
  print(f"global_size:  {prog_after.global_size}")
  print(f"local_size:   {prog_after.local_size}")
  print()
  print(prog_after.src)
