# Correctness check: fla's chunk_gated_delta_rule (tilelang backward on Hopper) vs the transformers
# reference PyTorch implementation, same inputs. Shapes = Qwen3.5-4B linear-attention layer:
# 16 qk heads x 128, 32 v heads x 128 (GVA), T=1024.
import torch, logging
logging.basicConfig(level=logging.INFO)
import transformers.models.qwen3_5.modeling_qwen3_5 as M
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_gdr
ref = M.torch_chunk_gated_delta_rule.__wrapped__     # the plain torch function under the decorator
torch.manual_seed(0)
B, T, H, HV, K, V = 1, 1024, 16, 32, 128, 128
def mk():
    q = torch.randn(B, T, HV, K, device="cuda", dtype=torch.bfloat16)   # Qwen3.5 repeats qk heads to HV before the call
    k = torch.randn(B, T, HV, K, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, T, HV, V, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(B, T, HV, device="cuda", dtype=torch.float32) * 0.5
    beta = torch.rand(B, T, HV, device="cuda", dtype=torch.bfloat16)
    return [x.requires_grad_() for x in (q, k, v, g, beta)]
a = mk(); b = [x.detach().clone().requires_grad_() for x in a]
o1, _ = fla_gdr(*a[:3], g=a[3], beta=a[4], use_qk_l2norm_in_kernel=True)
o2, _ = ref(*b[:3], g=b[3], beta=b[4], use_qk_l2norm_in_kernel=True)
do = torch.randn_like(o1)
(o1 * do).sum().backward(); (o2.to(o1.dtype) * do).sum().backward()
def rel(x, y): return ((x.float() - y.float()).norm() / y.float().norm()).item()
print("rel error  out", round(rel(o1, o2), 5), "| dq", round(rel(a[0].grad, b[0].grad), 5),
      "dk", round(rel(a[1].grad, b[1].grad), 5), "dv", round(rel(a[2].grad, b[2].grad), 5),
      "dg", round(rel(a[3].grad, b[3].grad), 5), "dbeta", round(rel(a[4].grad, b[4].grad), 5))
