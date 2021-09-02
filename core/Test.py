import torch
import numpy as np
# from core.loss import product, hamilton_product
# norm = torch.tensor([[1,1,1],[4,7,8],[9,7,3]], dtype=float)
# norm = norm**2
# sum  = torch.sum(norm, dim=1, keepdim=True)
# norm /= sum
# #print(norm)
# a = torch.zeros([2,4], dtype=float)
# a[0,0] = np.cos(10.0/180*3.14)
# a[0,1:] = np.sin(10.0/180*3.14)*norm[0]
#
# a[1, 0] = np.cos(0.7)
# a[1,1:] = np.sin(0.7)*norm[1]
#
# b = torch.tensor([[0,2,4,5],[1,3,5,6]], dtype=float)
# a = a.unsqueeze(0)
# b = b.unsqueeze(0)
#
# p1 = product(a,b)
# print(p1)
# p2 = hamilton_product(a,b)
# print(p2)

#
# from core.loss import Regularization
# test = Regularization()
# A = torch.rand([10,3,4], dtype=float)
# B = torch.rand([10,3,4], dtype=float)
# test(A,B)

from core.model import Net
net = Net(3,3,"bn")
A = torch.rand([2,1,32,32,32], dtype=torch.float)
result = net(A)
#print(net.fex)

