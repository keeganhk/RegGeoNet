import os, sys
prj_root = os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(prj_root)
from utils.pkgs import *
from utils.generic_cpnts import *
from utils.para_cpnts import *



class SONNBGClsParaLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, mode):
        assert mode in ['train', 'test']
        self.data_root = data_root
        self.mode = mode
        self.class_list = [line.strip() for line in open(os.path.join(data_root, 'class_list.txt'))]
        self.model_list = [line.strip() for line in open(os.path.join(data_root, mode + '_list.txt'))]
    def __getitem__(self, index):
        np.random.seed()
        model_name = self.model_list[index]
        class_name = model_name[:-5]
        cid = self.class_list.index(class_name)
        model_path = os.path.join(self.data_root, 'gi_obj_bg', class_name, model_name + '.npy')
        gi_pts = bounding_box_normalization(np.load(model_path).astype(np.float32))
        M = gi_pts.shape[0]
        m = int(np.sqrt(M))
        if self.mode == 'train':
            gi_pts = random_anisotropic_scaling(gi_pts, 2/3, 3/2)
            gi_pts = random_axis_rotation(gi_pts, 'z')
            gi_pts = random_translation(gi_pts, 0.2)
        gi_img = gi_pts.transpose().reshape(3, m, m)
        return gi_img, cid
    def __len__(self):
        return len(self.model_list)


class SONNONLYClsParaLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, mode):
        assert mode in ['train', 'test']
        self.data_root = data_root
        self.mode = mode
        self.class_list = [line.strip() for line in open(os.path.join(data_root, 'class_list.txt'))]
        self.model_list = [line.strip() for line in open(os.path.join(data_root, mode + '_list.txt'))]
    def __getitem__(self, index):
        np.random.seed()
        model_name = self.model_list[index]
        class_name = model_name[:-5]
        cid = self.class_list.index(class_name)
        model_path = os.path.join(self.data_root, 'gi_obj_only', class_name, model_name + '.npy')
        gi_pts = bounding_box_normalization(np.load(model_path).astype(np.float32))
        M = gi_pts.shape[0]
        m = int(np.sqrt(M))
        if self.mode == 'train':
            gi_pts = random_anisotropic_scaling(gi_pts, 2/3, 3/2)
            gi_pts = random_axis_rotation(gi_pts, 'z')
            gi_pts = random_translation(gi_pts, 0.2)
        gi_img = gi_pts.transpose().reshape(3, m, m)
        return gi_img, cid
    def __len__(self):
        return len(self.model_list)


class ResConv2D_V2(nn.Module):
    def __init__(self, in_channels, out_channels, leaky_slope):
        super(ResConv2D_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False), 
            nn.BatchNorm2d(out_channels)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False), 
            nn.BatchNorm2d(out_channels)
        )
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
        self.nl = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
    def forward(self, in_ftr):
        out_ftr = self.conv_2(self.nl(self.conv_1(in_ftr)))
        if self.in_channels != self.out_channels:
            out_ftr = self.nl(self.shortcut(in_ftr) + out_ftr)
        else:
            out_ftr = self.nl(in_ftr + out_ftr)
        return out_ftr


class ResSMLP_V2(nn.Module):
    def __init__(self, in_channels, out_channels, leaky_slope):
        super(ResSMLP_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.smlp_1 = SMLP(in_channels, out_channels, True, 'none')
        self.smlp_2 = SMLP(out_channels, out_channels, True, 'none')
        if in_channels != out_channels:
            self.shortcut = SMLP(in_channels, out_channels, True, 'none')
        self.nl = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
    def forward(self, in_ftr):
        out_ftr = self.smlp_2(self.nl(self.smlp_1(in_ftr)))
        if self.in_channels != self.out_channels:
            out_ftr = self.nl(self.shortcut(in_ftr) + out_ftr)
        else:
            out_ftr = self.nl(in_ftr + out_ftr)
        return out_ftr
    
    
class EdgeConvLayer(nn.Module):
    def __init__(self, Ci, Co, K, leaky_slope):
        super(EdgeConvLayer, self).__init__()
        self.Ci = Ci
        self.Co = Co
        self.K = K
        self.smlp = ResSMLP_V2(Ci*2, Co, leaky_slope)
    def forward(self, in_img_ftr):
        B, H, W = in_img_ftr.size(0), in_img_ftr.size(2), in_img_ftr.size(3)
        device = in_img_ftr.device
        N = H * W
        pwf = in_img_ftr.view(B, -1, N).permute(0, 2, 1).contiguous()
        Ci, Co, K = self.Ci, self.Co, self.K
        knn_idx = knn_search(pwf.detach(), pwf.detach(), K+1)[:, :, 1:]
        ftr_d = pwf.unsqueeze(2).repeat(1, 1, K, 1)
        ftr_n = index_points(pwf, knn_idx)
        ftr_e = torch.cat((ftr_d, ftr_n - ftr_d), dim=-1)
        ftr_e_updated = self.smlp(ftr_e.view(B, N*K, -1)).view(B, N, K, -1)
        ftr_a = torch.max(ftr_e_updated, dim=2)[0]
        out_img_ftr = ftr_a.permute(0, 2, 1).contiguous().view(B, -1, H, W)
        return out_img_ftr
    
    
class RegGeoNetClsSparseEncoder(nn.Module):
    def __init__(self, M, R, leaky_slope):
        super(RegGeoNetClsSparseEncoder, self).__init__()
        self.M = M
        self.R = R
        is_square_number(M)
        is_square_number(R)
        m = int(np.sqrt(M))
        r = int(np.sqrt(R))
        self.m = m
        self.r = r
        self.ibm_conv_1 = nn.Sequential(CU(6, 64, 1, True, 'leakyrelu', leaky_slope), ResConv2D_V2(64, 256, leaky_slope))
        self.cbm_conv_1 = EdgeConvLayer(256, 256, 8, leaky_slope)
        self.cbm_conv_2 = EdgeConvLayer(256, 512, 4, leaky_slope)
    def forward(self, img):
        bs = img.size(0)
        assert img.size(2) == img.size(3)
        ftr_0 = img
        ctr_p_0, ctr_i_0 = self.compute_ftr_ctr(ftr_0, self.r)
        rel_0 = ftr_0 - ctr_i_0
        cat_0 = torch.cat((ftr_0, rel_0), dim=1)
        ftr_1 = F.max_pool2d(self.ibm_conv_1(cat_0), self.r)
        ftr_2 = F.max_pool2d(self.cbm_conv_1(ftr_1), 2)
        ftr_3 = self.cbm_conv_2(ftr_2)
        vec_1 = F.adaptive_avg_pool2d(ftr_1, (1, 1)).squeeze(-1).squeeze(-1)
        vec_2 = F.adaptive_avg_pool2d(ftr_2, (1, 1)).squeeze(-1).squeeze(-1)
        vec_3 = F.adaptive_avg_pool2d(ftr_3, (1, 1)).squeeze(-1).squeeze(-1)
        ftr_f = torch.cat((vec_1, vec_2, vec_3), dim=-1)
        return ftr_f
    def compute_ftr_ctr(self, in_ftr, ss):
        bs, in_channels, height, width = in_ftr.size()
        assert np.mod(height, ss)==0 and np.mod(width, ss)==0
        ftr_pooled = F.avg_pool2d(in_ftr, ss)
        ftr_interp = F.interpolate(ftr_pooled, scale_factor=ss, mode='nearest')
        return ftr_pooled, ftr_interp
    
    
class RegGeoNetClsSparse(nn.Module):
    def __init__(self, num_classes):
        super(RegGeoNetClsSparse, self).__init__()
        self.enc = RegGeoNetClsSparseEncoder(256, 25, 0.2)
        fc_1 = nn.Sequential(nn.Linear(1024, 512, bias=False), nn.LeakyReLU(0.2, True), nn.Dropout(0.5))
        fc_2 = nn.Sequential(nn.Linear(512, 256, bias=False), nn.LeakyReLU(0.2, True), nn.Dropout(0.5))
        fc_3 = nn.Linear(256, num_classes, bias=False)
        self.cls = nn.Sequential(fc_1, fc_2, fc_3)
    def forward(self, img):
        cdw = self.enc(img)
        lgt = self.cls(cdw)
        return lgt


