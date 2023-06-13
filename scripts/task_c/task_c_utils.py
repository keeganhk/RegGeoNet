import os, sys
prj_root = os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(prj_root)
from utils.pkgs import *
from utils.generic_cpnts import *
from utils.para_cpnts import *



class MN40ClsParaLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, mode):
        assert mode in ['train', 'test']
        self.data_root = data_root
        self.mode = mode
        self.h5_file_path = os.path.join(data_root, 'mn40_para_' + mode + '.h5')
        self.class_list = [line.strip() for line in open(os.path.join(data_root, 'class_list.txt'))]
        with h5py.File(self.h5_file_path, 'r') as f:
            self.num_models = f['data'].shape[0]
    def __getitem__(self, index):
        np.random.seed()
        with h5py.File(self.h5_file_path, 'r') as f:
            para_data = f['data'][index].astype(np.float32)
            cid = f['label'][index]
        gi_pts = para_data[:, 0:3]
        M = gi_pts.shape[0]
        m = int(np.sqrt(M))
        if self.mode == 'train':
            gi_pts = bounding_box_normalization(random_anisotropic_scaling(gi_pts, 2/3, 3/2))
            gi_pts = random_translation(gi_pts, 0.2)
        gi_img = gi_pts.transpose().reshape(3, m, m)
        return gi_img, cid
    def __len__(self):
        return self.num_models


class EdgeConvLayer(nn.Module):
    def __init__(self, K, Ci, Co, lr_slope):
        super(EdgeConvLayer, self).__init__()
        self.K = K
        self.Ci = Ci
        self.Co = Co
        self.smlp = SMLP(Ci*2, Co, is_bn=True, nl='leakyrelu', slope=lr_slope)
    def forward(self, pwf):
        B, Ci, H, W = pwf.size()
        N = int(H * W)
        K = self.K
        Co = self.Co
        device = pwf.device
        pwf = pwf.view(B, Ci, N).permute(0, 2, 1).contiguous()
        knn_idx = knn_search(pwf.detach(), pwf.detach(), K+1)[:, :, 1:]
        ftr_d = pwf.unsqueeze(2).repeat(1, 1, K, 1)
        ftr_n = index_points(pwf, knn_idx)
        ftr_e = torch.cat((ftr_d, ftr_n - ftr_d), dim=-1)
        ftr_e_updated = self.smlp(ftr_e.view(B, N*K, -1)).view(B, N, K, -1)
        ftr_a = torch.max(ftr_e_updated, dim=2)[0]
        return ftr_a.permute(0, 2, 1).contiguous().view(B, -1, H, W)


class RegGeoNetClsEncoder(nn.Module):
    def __init__(self):
        super(RegGeoNetClsEncoder, self).__init__()
        self.conv_1 = CU(3+3, 64, 1, True, 'leakyrelu', 0.2)
        self.conv_2 = CU(3+3+64, 128, 1, True, 'leakyrelu', 0.2)
        self.conv_3 = CU(128+128, 256, 1, True, 'leakyrelu', 0.2)
        self.graph_conv_0 = EdgeConvLayer(20, 3, 128, 0.2)
        self.graph_conv_1 = EdgeConvLayer(10, 256, 256, 0.2)
        self.graph_conv_2 = EdgeConvLayer(5, 256, 512, 0.2)
        self.gap = nn.AdaptiveAvgPool2d(1)
    def forward(self, G):
        G_AP4 = F.avg_pool2d(G, kernel_size=4)
        G_AP4_NI4 = F.interpolate(G_AP4, scale_factor=4, mode='nearest')
        F1 = F.max_pool2d(self.conv_1(torch.cat((G, G-G_AP4_NI4), dim=1)), kernel_size=4)
        G_AP16 = F.avg_pool2d(G_AP4, kernel_size=4)
        G_AP16_NI4 = F.interpolate(G_AP16, scale_factor=4, mode='nearest')
        F2 = F.max_pool2d(self.conv_2(torch.cat((G_AP4, G_AP4-G_AP16_NI4, F1), dim=1)), kernel_size=4)
        F3 = self.conv_3(torch.cat((self.graph_conv_0(G_AP16), F2), dim=1))
        F4 = self.graph_conv_1(F.max_pool2d(F3, kernel_size=2))
        F5 = self.graph_conv_2(F.max_pool2d(F4, kernel_size=2))
        F3_ap  = self.gap(F3).squeeze(-1).squeeze(-1)
        F4_ap  = self.gap(F4).squeeze(-1).squeeze(-1)
        F5_ap  = self.gap(F5).squeeze(-1).squeeze(-1)
        F_cdw = torch.cat((F3_ap, F4_ap, F5_ap), dim=-1)
        return F_cdw


class ClsHead(nn.Module):
    def __init__(self, Ci, Nc):
        super(ClsHead, self).__init__()
        self.Ci = Ci
        self.Nc = Nc
        head_dims = [Ci, 512, 128, Nc]
        linear_1 = nn.Linear(head_dims[0], head_dims[1], bias=False)
        bn_1 = nn.BatchNorm1d(head_dims[1])
        nl_1 = nn.LeakyReLU(True, 0.2)
        dp_1 = nn.Dropout(0.5)
        self.fc_1 = nn.Sequential(linear_1, bn_1, nl_1, dp_1)
        linear_2 = nn.Linear(head_dims[1], head_dims[2], bias=False)
        bn_2 = nn.BatchNorm1d(head_dims[2])
        nl_2 = nn.LeakyReLU(True, 0.2)
        dp_2 = nn.Dropout(0.5)
        self.fc_2 = nn.Sequential(linear_2, bn_2, nl_2, dp_2)
        self.fc_3 = nn.Linear(head_dims[2], head_dims[3], bias=False)
    def forward(self, cdw):
        Ci, Nc = self.Ci, self.Nc
        B, D, device = cdw.size(0), cdw.size(1), cdw.device
        logits = self.fc_3(self.fc_2(self.fc_1(cdw)))
        return logits  


class RegGeoNetCls(nn.Module):
    def __init__(self, num_classes):
        super(RegGeoNetCls, self).__init__()
        self.encoder = RegGeoNetClsEncoder()
        self.head = ClsHead(1024, num_classes)
    def forward(self, G):
        cdw = self.encoder(G)
        logits = self.head(cdw)
        return logits



