'''This code demonstrates that before self-attention, the model can only combine previous tokens by averaging them equally. 
Three increasingly efficient ways to compute that average and then reveals the key insight: self-attention works by 
replacing equal averaging with learned importance weights, allowing the model to decide which previous tokens matter
 more for predicting the next token.'''


import torch
import torch.nn.functional as F

torch.manual_seed(42)
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

print("=" * 50)
print("VERSION 1: Naive loop")
print("=" * 50)
xbow1 = torch.zeros(B, T, C)
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]          # all tokens up to t
        xbow1[b, t] = xprev.mean(0)  # average them

print("Shape:", xbow1.shape)

print("\n" + "=" * 50)
print("VERSION 2: Matrix multiply with tril")
print("=" * 50)
tril = torch.tril(torch.ones(T, T))
wei  = tril / tril.sum(dim=1, keepdim=True)   # normalize rows
xbow2 = wei @ x
print("Max difference from v1:", (xbow1 - xbow2).abs().max().item())
print("Identical to v1:", torch.allclose(xbow1, xbow2))

print("\n" + "=" * 50)
print("VERSION 3: Softmax trick (data-independent weights)")
print("=" * 50)
tril = torch.tril(torch.ones(T, T))
wei  = torch.zeros(T, T)
wei  = wei.masked_fill(tril == 0, float('-inf'))
wei  = F.softmax(wei, dim=-1)
xbow3 = wei @ x
print("Weights (first row should be 1.0, rest equal):")
print(wei[0])
print(wei[3])
print("Identical to v2:", torch.allclose(xbow2, xbow3))

print("\n" + "=" * 50)
print("KEY INSIGHT")
print("=" * 50)
print("Right now weights are UNIFORM — every past token matters equally.")
print("Self-attention replaces the zeros with LEARNED Q@K dot products.")
print("That's the entire secret.")