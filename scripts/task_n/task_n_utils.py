import os, sys
prj_root = os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(prj_root)
from utils.pkgs import *
from utils.generic_cpnts import *
from utils.para_cpnts import *



class MN40RegParaLoader(torch.utils.data.Dataset):
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
        M = para_data.shape[0]
        m = int(np.sqrt(M))
        if self.mode == 'train':
            para_data = bounding_box_normalization(random_anisotropic_scaling(para_data, 2/3, 3/2))
            para_data = random_axis_rotation(para_data, 'z')
            para_data = random_translation(para_data, 0.20)
        gi_points = para_data[:, 0:3]
        gi_normals = para_data[:, 3:6]
        gi_points_reshaped = gi_points.transpose().reshape(3, m, m)
        gi_normals_reshaped = gi_normals.transpose().reshape(3, m, m)
        return gi_points_reshaped, gi_normals_reshaped
    def __len__(self):
        return self.num_models


class EdgeConvLayer2(nn.Module):
    def __init__(self, K, Ci, Co, lr_slope):
        super(EdgeConvLayer2, self).__init__()
        self.K = K
        self.Ci = Ci
        self.Co = Co
        smlp_1 = SMLP(Ci*2, Co, is_bn=True, nl='leakyrelu', slope=lr_slope)
        smlp_2 = SMLP(Co, Co, is_bn=True, nl='leakyrelu', slope=lr_slope)
        self.smlp = nn.Sequential(smlp_1, smlp_2)
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


class RegGeoNetReg(nn.Module):
    def __init__(self):
        super(RegGeoNetReg, self).__init__()
        self.conv_0 = CU(3, 16, 1, True, 'leakyrelu', 0.2)
        self.conv_1 = CU(3+3+16, 32, 1, True, 'leakyrelu', 0.2)
        self.conv_2 = CU(3+3+32, 64, 1, True, 'leakyrelu', 0.2)
        self.conv_3 = CU(64+64, 64, 1, True, 'leakyrelu', 0.2)
        self.graph_conv_0 = EdgeConvLayer2(20, 3, 64, 0.2)
        self.graph_conv_1 = EdgeConvLayer2(10, 64, 64, 0.2)
        self.graph_conv_2 = EdgeConvLayer2(10, 64, 128, 0.2)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.deconv_5 = CU(128+64, 256, 1, True, 'leakyrelu', 0.2)
        self.deconv_4 = CU(256+64+64, 128, 1, True, 'leakyrelu', 0.2)
        self.deconv_3 = CU(128+32, 64, 1, True, 'leakyrelu', 0.2)
        self.deconv_2 = CU(64+16, 64, 1, True, 'leakyrelu', 0.2)
        out_conv_1 = nn.Sequential(CU(64, 64, 1, True, 'leakyrelu', 0.2), nn.Dropout(p=0.5))
        out_conv_2 = nn.Sequential(CU(64, 64, 1, True, 'leakyrelu', 0.2), nn.Dropout(p=0.5))
        out_conv_3 = CU(64, 32, 1, True, 'leakyrelu', 0.2)
        out_conv_4 = CU(32, 3, 1, False, 'none')
        self.head = nn.Sequential(out_conv_1, out_conv_2, out_conv_3, out_conv_4)
    def forward(self, G):
        B, device = G.size(0), G.device
        F0 = self.conv_0(G)
        
        G_AP4 = F.avg_pool2d(G, kernel_size=4)
        G_AP4_NI4 = F.interpolate(G_AP4, scale_factor=4, mode='nearest')
        F1 = F.max_pool2d(self.conv_1(torch.cat((F0, G, G-G_AP4_NI4), dim=1)), kernel_size=4)
        
        G_AP16 = F.avg_pool2d(G_AP4, kernel_size=4)
        G_AP16_NI4 = F.interpolate(G_AP16, scale_factor=4, mode='nearest')
        F2 = F.max_pool2d(self.conv_2(torch.cat((G_AP4, G_AP4-G_AP16_NI4, F1), dim=1)), kernel_size=4)
        F3 = self.conv_3(torch.cat((self.graph_conv_0(G_AP16), F2), dim=1))
        
        F4 = self.graph_conv_1(F.max_pool2d(F3, kernel_size=2))
        F5 = self.graph_conv_2(F4)
        
        D5 = self.deconv_5(torch.cat((F5, F4), dim=1))
        D4 = self.deconv_4(torch.cat((self.up2(D5), F3, F2), dim=1))
        D3 = self.deconv_3(torch.cat((self.up4(D4), F1), dim=1))
        D2 = self.deconv_2(torch.cat((self.up4(D3), F0), dim=1))
        normals_pr = self.head(D2)
        return normals_pr




