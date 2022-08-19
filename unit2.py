import torch
import numpy as np

# Initializing a Tensor
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
print(x_data)

# From a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print(f"Numpy np_array value: \n {np_array} \n")
print(f"Tensor x_np value: \n {x_np} \n")

np.multiply(np_array, 2, out=np_array)

print(f"Numpy np_array after * 2 operation: \n {np_array} \n")
print(f"Tensor x_np value after modifying numpy array: \n {x_np} \n")

# From another tensor:
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# With random or constant values:
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# Attributes of a Tensor
tensor = torch.rand(3,4)

print(f"\nShape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}\n ")

# Operations on Tensors

# We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')

# Standard numpy-like indexing and slicing:
tensor = torch.ones(4, 4)
print('First row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(f"\nprint tensor:\n {tensor}\n")

# Joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
print(f"\ncheck 1: print tensor\n {tensor}\n")
print(f"\ncheck 2: print tensor.T\n {tensor.T}\n")    # transpose
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

print(f"\ncheck 3: print y1\n {y1}\n")
print(f"\ncheck 4: print y2\n {y2}\n")

y3 = torch.rand_like(tensor)
print(f"\ncheck 5: print y3\n {y3}\n")
torch.matmul(tensor, tensor.T, out=y3)
print(f"\ncheck 6: print y3\n {y3}\n")

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)
print(f"\ncheck 7: print tensor\n {tensor}\n")
print(f"\ncheck 8: print z1\n {z1}\n")
print(f"\ncheck 9: print z2\n {z2}\n")

z3 = torch.rand_like(tensor)
print(f"\ncheck 10: print z3\n {z3}\n")
torch.mul(tensor, tensor, out=z3)
print(f"\ncheck 11: print z3\n {z3}\n")

# Single-element tensors
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# in-place operation
print(tensor, "\n")
tensor.add_(5)
print(tensor)

# Bridge with NumPy
# Tensor to NumPy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
print(" ")

# NumPy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)
print(f"n: {n}")
print(f"t: {t}")

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")