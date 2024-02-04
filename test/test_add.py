import torch
from driss_torch import add_one

shape = (3, 3, 3)
a = torch.randint(0, 10, shape, dtype=torch.float).cuda()

print(a)
print(add_one(a))
