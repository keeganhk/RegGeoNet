import os, sys
prj_root = os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(prj_root)
from utils.pkgs import *
from utils.generic_cpnts import *
from utils.para_cpnts import *

from matplotlib import cm




class SNPSegParaLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, mode):
        assert mode in ['train', 'test']
        self.data_root = data_root
        self.mode = mode
        self.class_list = [line.strip() for line in open(os.path.join(data_root, 'class_list.txt'), 'r')] 
        self.model_list = [line.strip() for line in open(os.path.join(data_root, mode + '_list.txt'), 'r')]
        self.num_models = len(self.model_list)
    def __getitem__(self, index):
        np.random.seed()
        model_name = self.model_list[index]
        class_name = model_name[:-5]
        cid = self.class_list.index(class_name)
        para_data = np.load(os.path.join(self.data_root, 'gi_labeled', class_name, model_name + '.npy'))
        M = para_data.shape[0]
        m = int(np.sqrt(M))
        gi_pts = para_data[:, 0:3].astype(np.float32)
        gi_lbs = para_data[:, 3:4].astype(np.int64)
        if self.mode == 'train':
            gi_pts = random_anisotropic_scaling(gi_pts, 2/3, 3/2)
        gi_pts_reshaped = gi_pts.transpose().reshape(3, m, m)
        gi_lbs_reshaped = gi_lbs.transpose().reshape(1, m, m)
        return gi_pts_reshaped, gi_lbs_reshaped, cid, model_name
    def __len__(self):
        return self.num_models


def ShapeNetPart_ObjectsParts():
    # 16 different objects, 50 different parts
    objects_names = [
        'airplane', 
        'bag', 
        'cap', 
        'car', 
        'chair', 
        'earphone', 
        'guitar', 
        'knife', 
        'lamp', 
        'laptop', 
        'motorbike', 
        'mug', 
        'pistol', 
        'rocket', 
        'skateboard', 
        'table'
    ]
    objects_parts = {
        'airplane': [0, 1, 2, 3], 
        'bag': [4, 5], 
        'cap': [6, 7], 
        'car': [8, 9, 10, 11], 
        'chair': [12, 13, 14, 15], 
        'earphone': [16, 17, 18], 
        'guitar': [19, 20, 21], 
        'knife': [22, 23],
        'lamp': [24, 25, 26, 27], 
        'laptop': [28, 29], 
        'motorbike': [30, 31, 32, 33, 34, 35], 
        'mug': [36, 37], 
        'pistol': [38, 39, 40], 
        'rocket': [41, 42, 43], 
        'skateboard': [44, 45, 46], 
        'table': [47, 48, 49]}
    return objects_names, objects_parts


def ShapeNetPart_PartsColors():
    # parts_colors: (50, 3), color-code each of the 50 different parts
    objects_names, objects_parts = ShapeNetPart_ObjectsParts()
    num_parts = []
    for k, v in objects_parts.items():
        num_parts.append(len(v))
    cmap = cm.jet
    parts_colors = np.zeros((50, 3))
    i = 0
    for num in num_parts:
        base_colors = cmap(np.linspace(0, 1, num))[:, 0:3]
        for k in range(num):
            parts_colors[i, ...] = base_colors[k, ...]
            i += 1
    return parts_colors


def ShapeNetPart_ColorCode(points_with_labels):
    # points_with_labels: [num_points, 4]
    assert points_with_labels.ndim==2 and points_with_labels.size(-1)==4
    points = points_with_labels[:, 0:3].unsqueeze(0) # [1, num_points, 3]
    labels = points_with_labels[:, -1].unsqueeze(0).long() # [1, num_points]
    parts_colors =  torch.tensor(ShapeNetPart_PartsColors()).unsqueeze(0).to(points_with_labels.device) # [1, 50, 3]
    color_codes = index_points(parts_colors, labels) # [1, num_points, 3]
    points_color_coded = torch.cat((points, color_codes), dim=-1).squeeze(0) # [num_points, 6]
    return points_color_coded


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


class RegGeoNetSeg(nn.Module):
    def __init__(self, num_object_classes, num_part_classes):
        super(RegGeoNetSeg, self).__init__()
        self.num_object_classes = num_object_classes
        self.num_part_classes = num_part_classes
        self.conv_0 = CU(3, 16, 1, True, 'leakyrelu', 0.2)
        self.conv_1 = CU(3+3+16, 32, 1, True, 'leakyrelu', 0.2)
        self.conv_2 = CU(3+3+32, 64, 1, True, 'leakyrelu', 0.2)
        self.conv_3 = CU(64+64, 64, 1, True, 'leakyrelu', 0.2)
        self.graph_conv_0 = EdgeConvLayer2(20, 3, 64, 0.2)
        self.graph_conv_1 = EdgeConvLayer2(10, 64, 64, 0.2)
        self.graph_conv_2 = EdgeConvLayer2(10, 64, 128, 0.2)
        self.lift = FC(num_object_classes, 64, True, 'leakyrelu', 0.2)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.deconv_5 = CU(128+64+64, 256, 1, True, 'leakyrelu', 0.2)
        self.deconv_4 = CU(256+64+64, 128, 1, True, 'leakyrelu', 0.2)
        self.deconv_3 = CU(128+32, 64, 1, True, 'leakyrelu', 0.2)
        self.deconv_2 = CU(64+16, 64, 1, True, 'leakyrelu', 0.2)
        out_conv_1 = nn.Sequential(CU(64, 64, 1, True, 'leakyrelu', 0.2), nn.Dropout(p=0.5))
        out_conv_2 = nn.Sequential(CU(64, 64, 1, True, 'leakyrelu', 0.2), nn.Dropout(p=0.5))
        out_conv_3 = CU(64, 32, 1, True, 'leakyrelu', 0.2)
        out_conv_4 = CU(32, num_part_classes, 1, False, 'none')
        self.head = nn.Sequential(out_conv_1, out_conv_2, out_conv_3, out_conv_4)
    def forward(self, G, cid):
        assert G.size(0) == cid.size(0)
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
        cid_one_hot = F.one_hot(cid, self.num_object_classes).float().to(device)
        cid_lifted = self.lift(cid_one_hot)
        cid_lifted_exp = cid_lifted.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, F5.size(2), F5.size(3))
        D5 = self.deconv_5(torch.cat((F5, F4, cid_lifted_exp), dim=1))
        D4 = self.deconv_4(torch.cat((self.up2(D5), F3, F2), dim=1))
        D3 = self.deconv_3(torch.cat((self.up4(D4), F1), dim=1))
        D2 = self.deconv_2(torch.cat((self.up4(D3), F0), dim=1))
        logits = self.head(D2)
        return logits


