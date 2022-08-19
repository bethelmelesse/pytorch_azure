import torch
from torch import nn
print(" ")

x = torch.ones(5)                 # tensor([1., 1., 1., 1., 1.])
y = torch.zeros(3)                # tensor([0., 0., 0.])
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print('Gradient function for z =', z.grad_fn)
print('Gradient function for loss =', loss.grad_fn)

loss.backward()
print(w.grad)
print(b.grad)

print(" ")

z = torch.matmul(x, w)+ b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x,w) +b
print(z.requires_grad)

print(" ")

z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)

inp = torch.eye(5, requires_grad= True)
out= (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print(out)
print("First call\n", inp.grad)
out.backward(torch.ones_like(inp), retain_graph=True)
print("Second call\n", inp.grad)
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nCall after zeroing gradients\n", inp.grad)
print(" ")