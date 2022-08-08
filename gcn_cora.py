import scipy.sparse as sp
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F



def encode_onehot(labals):  #对输入的特征进行onehot编码
    classes=set(labals)     #创建无序不重复的序列
    class_dict={c: np.identity(len(classes))[i,:] for i,c in enumerate(classes)}#[i,:]相当于将第i行的所有列赋给i位置所在的元素
    #形如上面的写法，将参数定义好之后先运行for循环中的内容，再将所遍历出的内容赋给相应参数

    labal_onehot=np.array(list(map(class_dict.get,labals)),dtype=np.int32)

    return labal_onehot     #得到每个labals对应的onehot向量

def nomalize(mx):#传入一个稀疏矩阵，（进行归一化的原因）采用加法规则时，对于度大的节点特征越来越大，而对于度小的节点却相反，这可能导致网络训练过程中梯度爆炸或者消失的问题。
    #这种聚合方式实际上就是在对邻接求和取平均，属于非对称的归一化方式
    rownum=np.array(mx.sum(1))#计算传入矩阵的行和,并且变成1维的array
    r_inv=np.power(rownum,-1).flatten()
    r_inv[np.isinf(r_inv)]=0#由于上边进行了求倒数，故可能出现无穷大的情况，所以这里将无穷大变成了0
    r_mat_inv=sp.diags(r_inv)#随后将其变成对角线的形式，这里就是图卷积中的度矩阵D^-1
    mx=r_mat_inv.dot(mx)

    return mx

def accuracy(output,labels):
    pred=output.max(1)[1].type_as(labels)
    correct=pred.eq(labels).double()
    correct=correct.sum()

    return correct/len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx=sparse_mx.tocoo().astype(np.float32)
    indices=torch.from_numpy(np.vstack((sparse_mx.row,sparse_mx.col)).astype(np.int64))#np.vstack按垂直方向堆叠数组
    #稀疏矩阵只会记录有值的位置上的坐标和元素，sparse_mx.row就是存在元素的行坐标，sparse_mx.col就是存在元素所在列的坐标，将其
    #堆叠成垂直方向的数组
    values=torch.from_numpy(sparse_mx.data)#rom_numpy将生成数组变成转换为张量类型，sparse_mx.data取到稀疏矩阵的元素
    shape=torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices,values,shape)

def load_data(path="data/cora/",dataset="cora"):
    print("Loading data...")

    idx_features_labels=np.genfromtxt("{}{}.content".format(path,dataset),dtype=np.dtype(str))#从文本文件加载数据，并按指定处理缺失值
    features=sp.csr_matrix(idx_features_labels[:,1:-1],dtype=np.float32)
    labels=encode_onehot(idx_features_labels[:,-1])
    #build graph
    idx=np.array(idx_features_labels[:,0],dtype=np.int32)
    idx_map={j:i for i,j in enumerate(idx)}
    edges_unordered=np.genfromtxt("{}{}.cites".format(path,dataset),dtype=np.int32)#cites中分别是（被引用论文的编号）+（引用论文的编号）
    # a=edges_unordered.flatten()
    # c=edges_unordered.shape
    edges=np.array(list(map(idx_map.get,edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)#edges_unordered.shape=(5429,2)
    # b=idx_map[1033]
    shapeee=np.ones(edges.shape[0])
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0],labels.shape[0]),
                        dtype=np.float32)


    adj=adj+adj.T.multiply(adj.T > adj)-adj.multiply(adj.T > adj)

    features=nomalize(features)
    # a=sp.eye(adj.shape[0])
    adj=nomalize(adj+sp.eye(adj.shape[0]))#sp.eye生成对角线全是1的矩阵

    idx_train=range(140)
    idx_val=range(200,500)
    idx_test=range(500,1500)

    features=torch.FloatTensor(np.array(features.todense()))
    labels=torch.LongTensor(np.where(labels)[1])#cora只有7类，变成onehot编码之后就可以用编号0~6来代表类别了
    adj=sparse_mx_to_torch_sparse_tensor(adj)

    idx_train=torch.LongTensor(idx_train)
    idx_val=torch.LongTensor(idx_val)
    idx_test=torch.LongTensor(idx_test)
    print("数据加载成功...")

    return adj,features,labels,idx_train,idx_val,idx_test

class GraphConvolution(nn.Module):
    def __init__(self,in_features,out_features,bias=True):
        super(GraphConvolution,self).__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.weight=nn.Parameter(torch.FloatTensor(in_features,out_features))
        self.use_bias=bias
        if self.use_bias:
            self.bias=nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)#kaiming正态分布初始化卷积层参数
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self,input_feature,adj):
        support=torch.mm(input_feature,self.weight)
        output=torch.spmm(adj,support)
        if self.use_bias:
            return output+self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self,input_dim=1433):
        super(GCN,self).__init__()
        self.gnc1=GraphConvolution(input_dim,16)
        self.gcn2=GraphConvolution(16,7)
        pass

    def forward(self,X,adj):
        X=F.relu(self.gnc1(X,adj))
        X=self.gcn2(X,adj)

        return F.log_softmax(X,dim=1)

#gcn一般可以分成两个模块，第一个模块叫做GraphConvolution，是来进行图邻接矩阵等的处理和需要参数的初始化，在其内的forward中得到一个输出
#而后是GCN模块，主要类似于CNN的backbone，进行网络架构的设计，最终输出的结果就是每个节点的特征

adj, features, labels, idx_train, idx_val, idx_test = load_data()
model=GCN()#features.shape[1]
optimizer=optim.Adam(model.parameters(),lr=0.01,weight_decay=5e-4)

def train(epochs):
    for epoch in range(epochs):
        output=model(features,adj)
        loss_train=F.nll_loss(output[idx_train],labels[idx_train])
        #NLLLoss （负对数似然损失）函数输入 input 之前，需要对 input 进行 log_softmax 处理，即将 input 转换成概率分布的形式，并且取对数，底数为 e
        acc_train=accuracy(output[idx_train],labels[idx_train])
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        loss_val=F.nll_loss(output[idx_val],labels[idx_val])
        acc_val=accuracy(output[idx_val],labels[idx_val])

        if (epoch % 10 == 0):
            print("Epoch:{}".format(epoch+1),
                  "loss_train:{:.4f}".format(loss_train.item()),
                  "acc_train:{:.4f}".format(acc_train.item()),
                  "loss_val:{:.4f}".format(loss_val.item()),
                  "acc_val:{:.4f}".format(acc_val.item()))

if __name__=="__main__":
    train(200)