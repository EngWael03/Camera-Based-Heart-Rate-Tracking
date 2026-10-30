import torch

x = torch.tensor([1.0], requires_grad=True)
y = torch.tensor([2.0], requires_grad=True)

f = 2 * (x**3) * y + 4 * x * (y**2)

f.backward()

print(f"f(x,y) = {f.item()}")
print(f"derivative w.r.t. x = {x.grad.item()}")
print(f"derivative w.r.t. y = {y.grad.item()}")