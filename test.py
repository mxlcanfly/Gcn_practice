import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import coo_matrix


# a = np.array([[1,0,0,1,0],[0,1,1,0,0],[0,0,0,0,1],[1,0,0,0,1],[0,0,1,0,0]])
# a = sp.coo_matrix(a,shape=(5,5))
# sparse_mx = a.tocoo().astype(np.float32)
# print(sparse_mx.row)
# print(sparse_mx.col)
# print(sparse_mx.data)
# print(sparse_mx.shape)
# indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)) # 存储非零元素的行列坐标
# print(indices)
# values = torch.from_numpy(sparse_mx.data)#
# print(values)
# shape = torch.Size(sparse_mx.shape)
# print(shape)
# print(torch.sparse.FloatTensor(indices, values, shape))


# adj = coo_matrix((np.ones(5), ([3, 4, 0, 2, 1], [0, 2, 1, 4, 3])), shape=(5, 5), dtype=np.float32)
# adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)#将非对称阵变成对称阵
# print(adj)
# a=np.array([['1','0','0','0','lllll'],['2','0','0','0','xxxxx']])
# print(a)
# b=a[1:,1:-1]
# print(b)
# d={2:1,1:3,1:4,5:7}
# a=np.array([[1,2],[1,5]])
# s=a.shape
# print(s)
# b=np.array(list(map(d.get,a.flatten()))).reshape(a.shape)
# print(b)

a=np.array([[1,2,3],[4,5,6]])
b=torch.FloatTensor(np.where(a)[1])
print(b)