import torch
import torch.nn as nn
import torch.nn.functional as F

class Regularization(nn.Module):
    def __init__(self, device):
        super(Regularization, self).__init__()
        self.Id = torch.eye(3, requires_grad=False).unsqueeze(0).to(device)
    def forward(self, plane, rot):
        plane = [p.unsqueeze(1) for p in plane]
        plane = torch.cat(plane, dim=1)

        rot = [r.unsqueeze(1) for r in rot]
        rot = torch.cat(rot, dim=1)
        #print(rot.shape, plane.shape)
        M1 = plane[:, :, :-1]
        M1 = F.normalize(M1, dim=2)
        M2 = rot[:, :, 1:]
        M2 = F.normalize(M2, dim=2)
        A = torch.matmul(M1, torch.transpose(M1,1,2)) - self.Id
        B = torch.matmul(M2, torch.transpose(M2,1,2)) - self.Id
        return torch.sum((A**2 + B**2))

def product(q1, q2):
    b,n,_ = q1.shape
    q1 = q1.view(-1,1,4)
    q2 = q2.view(-1,1,4)
    p = torch.zeros([b*n,4]).to(q1.device)
    p[:,0] = q1[:,0,0] * q2[:,0,0] - torch.matmul(q1[:,:,1:], torch.transpose(q2[:,:,1:],1,2))[:,0,0]
    p[:,1:] = q1[:,0,:1].repeat(1,1,3) * q2[:,0,1:] +\
              q1[:,0,1:]*q2[:,0,:1].repeat(1,1,3) + \
                torch.cross(q1[:,:,1:], q2[:,:,1:])[:,0,:]
    return p.reshape(b,n,4)

def quat_conjugate(quat):
  q_conj = quat
  q_conj[:, :, 1:] *= -1
  return q_conj

class DistanceLoss(nn.Module):
    def __init__(self, samples, device):
        super(DistanceLoss, self).__init__()
        self.N = samples
        self.valid = False
        self.device = device
    def forward(self, samples, close_point, voxel, plane, rot):
        plane_loss = torch.Tensor([0]).to(self.device)
        rot_loss = torch.Tensor([0]).to(self.device)
        #TODO 处理损失函数
        for p in plane :
            _points = self.planeTrans(samples, p)
            inds = self.point2Ind(_points)
            targets = self.closePoins(close_point, inds)
            mask = self.mask(voxel, inds)
            plane_loss += self.distance(_points, targets, mask)

        for r in rot :
            _points = self.rotTrans(samples, r)
            inds = self.point2Ind(_points)
            targets = self.closePoins(close_point, inds)
            mask = self.mask(voxel, inds)
            rot_loss += self.distance(_points, targets, mask)

        return plane_loss + rot_loss

    def planeTrans(self, point, plane):
        abc = plane[:, 0:3].unsqueeze(1).repeat(1, self.N, 1)
        d = plane[:, 3:].unsqueeze(1).repeat(1, self.N, 1)
        up = torch.sum(point*abc, 2, True) + d
        down  = torch.sum(abc**2, 2, True) + 1e-5
        factor = up/down*2
        new_point = point - factor*abc
        return new_point

    def rotTrans(self, point, quat):
        quat = quat.unsqueeze(1).repeat(1, self.N, 1)
        _zeros = torch.zeros_like(point[:,:, 0:1])
        quat_point = torch.cat([_zeros, point], dim=2)
        quatT = quat_conjugate(quat)
        new_point = product(quat, quat_point)
        new_point = product(new_point, quatT)
        return new_point[:,:,1:]

    def distance(self, preds, targets, mask):
        return torch.sum((preds - targets)**2*mask)

    @staticmethod
    def closePoins(closePoints, ind):
        ind = ind.unsqueeze(2).repeat(1,1,3).long()
        cp = torch.gather(closePoints, 1, ind)
        return cp
    @staticmethod
    def mask(voxel, ind):
        voxel = voxel.view(-1, 32*32*32)
        ind = ind.long()
        _voxel = torch.gather(voxel, 1, ind)
        mask = _voxel.unsqueeze(2)
        return mask

    @staticmethod
    def point2Ind(point):
        upBound = 0.5
        lowBound = -0.5
        inds = (point - lowBound) / (upBound - lowBound) * 32
        inds = torch.round(torch.clamp(inds, min=0, max=31))
        indx, indy, indz = inds[:,:,0], inds[:,:,1], inds[:, :,2]
        return indx*indy*indz

