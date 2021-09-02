import torch
import torch.nn as nn

class PlaneHead(nn.Module):
    def __init__(self, n, in_cn):
        super(PlaneHead, self).__init__()
        self.expect_n = n
        self.transfrom = []
        init_bias = [[1,0,0,0], [0,1,0,0], [0,0,1,0]]
        for i in range(n):
            self.transfrom.append(nn.Sequential())
            self.transfrom[-1].add_module("Plane Linear-1-"+str(i), nn.Linear(in_cn, in_cn//2))
            self.transfrom[-1].add_module("activate Layer-1-"+str(i), nn.LeakyReLU())
            self.transfrom[-1].add_module("Plane Linear-2-"+str(i), nn.Linear(in_cn//2, in_cn//4))
            self.transfrom[-1].add_module("activate Layer-2-"+str(i), nn.LeakyReLU())
            finial = nn.Linear(in_cn//4, 4)
            finial.bias.data = torch.Tensor(init_bias[i])
            self.transfrom[-1].add_module("finial "+str(i), finial)

    def forward(self, feature):
        out = []
        for tran in self.transfrom :
            out.append(tran(feature))
        return out

class RotHead(nn.Module):
    def __init__(self, n, in_cn):
        super(RotHead, self).__init__()
        self.expect_n = n
        self.transfrom = []
        init_bias = [[0,1,0,0], [0,0,1,0], [0,0,0,1]]
        for i in range(n):
            self.transfrom.append(nn.Sequential())
            self.transfrom[-1].add_module("Plane Linear-1-" + str(i), nn.Linear(in_cn, in_cn // 2))
            self.transfrom[-1].add_module("activate Layer-1-", nn.LeakyReLU())
            self.transfrom[-1].add_module("Plane Linear-2-" + str(i), nn.Linear(in_cn // 2, in_cn // 4))
            self.transfrom[-1].add_module("activate Layer-2-", nn.LeakyReLU())
            finial = nn.Linear(in_cn // 4, 4)
            finial.weight.data *= 0
            finial.bias.data = torch.Tensor(init_bias[i])
            self.transfrom[-1].add_module("Plane Linear " + str(i), finial)
            #self.transfrom[-1].add_module("activate Layer", nn.LeakyReLU())

    def forward(self, feature):
        out = []
        for trans in self.transfrom:
            out.append(trans(feature))
        return out

class Net(nn.Module):
    def __init__(self, pn, rn, bn):
        super(Net, self).__init__()

        self.bn = self.selectBN(bn)
        self.fex = nn.Sequential()
        self.fex.add_module("[32, 16, 1, 4]-1", nn.Conv3d(1, 4,  (3,3,3), 1, 1))
        self.fex.add_module("[POOL, 2]-1", nn.MaxPool3d(2))
        self.fex.add_module("[BN]-1", self.bn(4))
        self.fex.add_module("[activate Leaky RELU]-1", nn.LeakyReLU())


        self.fex.add_module("[16, 4, 8]-2", nn.Conv3d(4, 8,  (3,3,3), 1, 1))
        self.fex.add_module("[POOL, 2]-2", nn.MaxPool3d(2))
        self.fex.add_module("[BN]-2", self.bn(8))
        self.fex.add_module("[activate Leaky RELU]-2", nn.LeakyReLU())


        self.fex.add_module("[8, 8, 16]-3", nn.Conv3d(8, 16, (3,3,3), 1, 1))
        self.fex.add_module("[POOL, 2]-3", nn.MaxPool3d(2))
        self.fex.add_module("[BN]-3", self.bn(16))
        self.fex.add_module("[activate Leaky RELU]-3", nn.LeakyReLU())

        #
        self.fex.add_module("[4, 16,32]-4", nn.Conv3d(16, 32,(3,3,3), 1, 1))
        self.fex.add_module("[POOL, 2]-4", nn.MaxPool3d(2))
        self.fex.add_module("[BN]-4", self.bn(32))
        self.fex.add_module("[activate Leaky RELU]-4", nn.LeakyReLU())


        self.fex.add_module("[2, 32,64]-5", nn.Conv3d(32, 64,(3,3,3), 1, 1))
        self.fex.add_module("[POOL, 2]-5", nn.MaxPool3d(2))
        self.fex.add_module("[BN]-5", self.bn(64))
        self.fex.add_module("[activate Leaky RELU]-5", nn.LeakyReLU())


        self.Plane = PlaneHead(pn, 64)
        self.Rot = RotHead(rn, 64)

    def forward(self, voxel):
        #print(voxel.shape)
        feature = self.fex(voxel)
        feature = feature.view(-1, 64)
        #print(feature.shape)
        Plane = self.Plane(feature)  # a, b,c,d  shape: [n,[b, 4]]
        Rot = self.Rot(feature)      # \pi x,y,z shape: [n,[b, 3]]

        return Plane, Rot

    def selectBN(self, name):
        if name == "bn":
            return torch.nn.BatchNorm3d
        elif name == "Lbn":
            return torch.nn.LayerNorm
        else:
            pass
        print("unexpected batchNorm layer in Net.selectBN")
        exit()