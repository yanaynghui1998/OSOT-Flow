import torch
import torch.nn as nn
import torch.nn.functional as F

def normalized_feature_distance(features_N, features_P, features_A, omega=[1/32,1/16,1/8,1/4], beta=1.0, eps=1e-8):

    total_loss = 0.0
    for i in range(len(omega)):
        f_J = features_P[i]#positive
        f_I = features_N[i]#negative
        f_pred = features_A[i]#archor

        # L2 距离（按样本平均）
        d1 = torch.norm((f_J - f_pred).view(f_J.size(0), -1), dim=1)  # shape [B]
        d2 = torch.norm((f_I - f_pred).view(f_I.size(0), -1), dim=1)  # shape [B]

        # 避免除以 0，做归一化
        loss_i = omega[i] * (d1 / (d2))  # shape [B]
        total_loss += loss_i.mean()

    return beta * total_loss

def normalized_feature_distance_l1(features_N, features_P, features_A, omega=[1/32,1/16,1/8,1/4], beta=1.0, eps=1e-8):
    total_loss = 0.0
    for i in range(len(omega)):
        f_J = features_P[i]
        f_I = features_N[i]
        f_pred = features_A[i]

        # L1 距离（按样本平均）
        d1 = (f_J - f_pred).view(f_J.size(0), -1).abs().sum(dim=1)  # shape [B]
        d2 = (f_I - f_pred).view(f_I.size(0), -1).abs().sum(dim=1)  # shape [B]

        # 归一化，避免除 0
        loss_i = omega[i] * (d1 / (d2 + eps))
        total_loss += loss_i.mean()

    return beta * total_loss

class ContrastLoss(nn.Module):
    def __init__(self, ablation=True):
        super(ContrastLoss, self).__init__()
        
        self.l1 = nn.L1Loss()
        self.l2=nn.MSELoss(reduction='mean')  
        self.weights = [1.0, 1.0, 1.0, 1.0]
        self.ab = ablation
    def forward(self, features_N, features_P, features_A):
        loss = 0
        for i in range(len(features_N)):
            d_ap = self.l2(features_A[i], features_P[i].detach())  # L2(anchor, positive)
            if not self.ab:
                d_an = self.l2(features_A[i], features_N[i].detach())  # L2(anchor, negative)
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap
            loss += self.weights[i] * contrastive
        loss=loss/len(features_A)
        return loss


class ContrastLoss_Grad(nn.Module):
    def __init__(self, ablation=True):
        super(ContrastLoss_Grad, self).__init__()
        self.weights = [1.0/32, 1.0/32, 1.0/32, 1.0/32]
        self.ab = ablation
    def forward(self, features_N, features_P, features_A):
        loss = 0
        for i in range(len(features_N)):
            d_ap = (features_P[i].detach()-features_A[i]).abs().mean()  # L2(anchor, positive)
            if not self.ab:
                d_an = (features_N[i].detach()-features_A[i]).abs().mean()  # L2(anchor, negative)
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap
            loss += self.weights[i] * contrastive
        return loss


class HCRLoss3D(nn.Module):
    def __init__(self, weights=None, mode='l1'):
        super().__init__()
        self.mode = mode.lower()
        assert self.mode in ['l1', 'l2']

        self.dist_fn = nn.L1Loss(reduction='mean') if self.mode == 'l1' else nn.MSELoss(reduction='mean')
        self.weights = weights  # List[float]，每层的权重

    def forward(self, a_feat_list, p_feat_list, n_feat_list):
        assert len(a_feat_list) == len(p_feat_list) == len(n_feat_list)

        num_layers = len(a_feat_list)
        if self.weights is None:
            self.weights = [1.0] * num_layers

        total_loss = 0.0
        for i in range(num_layers):
            a = a_feat_list[i]              
            p = p_feat_list[i].detach()    
            n = n_feat_list[i].detach()     
            B, C, D, H, W = a.shape

            d_ap = self.dist_fn(a, p.detach())

            a_exp = a.unsqueeze(1).expand(B, B, C, D, H, W)     
            n_exp = n.expand(B, B, C, D, H, W)                  
            d_an = self.dist_fn(a_exp, n_exp.detach())                 

            contrastive = d_ap / (d_an + 1e-7)
            total_loss += self.weights[i] * contrastive
        # total_loss=total_loss/len(a_feat_list)
        return total_loss


class HCRLoss3D_Triplet(nn.Module):
    def __init__(self, weights=None, mode='l1'):
        super().__init__()
        self.mode = mode.lower()
        assert self.mode in ['l1', 'l2']

        self.dist_fn = torch.nn.TripletMarginLoss(margin=1.0, p=2)
        self.weights = weights  # List[float]，每层的权重

    def forward(self, a_feat_list, p_feat_list, n_feat_list):
        assert len(a_feat_list) == len(p_feat_list) == len(n_feat_list)

        num_layers = len(a_feat_list)
        if self.weights is None:
            self.weights = [1.0] * num_layers

        total_loss = 0.0
        for i in range(num_layers):
            a = a_feat_list[i]              
            p = p_feat_list[i].detach()    
            n = n_feat_list[i].detach()     
            contrastive = self.dist_fn(a, p.detach(),n.detach())
            total_loss += self.weights[i] * contrastive
        total_loss=total_loss/len(a_feat_list)
        return total_loss

class HCRLoss3D_Cosine(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights 

    def forward(self, a_feat_list, p_feat_list, n_feat_list):
        assert len(a_feat_list) == len(p_feat_list) == len(n_feat_list)

        num_layers = len(a_feat_list)
        if self.weights is None:
            self.weights = [1.0] * num_layers

        total_loss = 0.0
        eps = 1e-7

        for i in range(num_layers):
            a = a_feat_list[i]       
            p = p_feat_list[i].detach()
            n = n_feat_list[i].detach()
            B = a.shape[0]

            a_flat = a.view(B, -1)
            p_flat = p.view(B, -1)
            n_flat = n.view(B, -1)

            cos_ap = F.cosine_similarity(a_flat, p_flat, dim=1) 
            cos_an = F.cosine_similarity(a_flat, n_flat, dim=1)

            d_ap = 1 - cos_ap
            d_an = 1 - cos_an
            contrastive = d_ap / (d_an + eps)

            total_loss += self.weights[i] * contrastive.mean()

        return total_loss