from .pkgs import *
from .generic_cpnts import *



def build_lattice(H, W):
    h_p = np.linspace(-0.5, +0.5, H, dtype=np.float32)
    w_p = np.linspace(-0.5, +0.5, W, dtype=np.float32)
    grid_points = np.array(list(itertools.product(h_p, w_p)))
    h_i = np.linspace(0, H-1, H, dtype=np.int64)
    w_i = np.linspace(0, W-1, W, dtype=np.int64)
    grid_indices = np.array(list(itertools.product(h_i, w_i)))
    return grid_points, grid_indices


def build_lattice_01(H, W):
    h_p = np.linspace(0, 1, H, dtype=np.float32)
    w_p = np.linspace(0, 1, W, dtype=np.float32)
    grid_points = np.array(list(itertools.product(h_p, w_p)))
    h_i = np.linspace(0, H-1, H, dtype=np.int64)
    w_i = np.linspace(0, W-1, W, dtype=np.int64)
    grid_indices = np.array(list(itertools.product(h_i, w_i)))
    return grid_points, grid_indices


def anchor_sampling(points, num_samples, grid_subsample=False, grid_size=None):
    num_points = points.shape[0]
    assert points.ndim==2 and points.shape[1]==3
    if not grid_subsample:
        anchors = farthest_point_sampling_fix_first(points, num_samples)
    else:
        # grid_size=0.03 is a good choice for objects centralized & normalized into a sphere of [-1, +1] (radius=2)
        assert grid_size is not None
        points_gss = grid_subsample_cpu(torch.tensor(points), grid_size)
        num_points_gss = points_gss.shape[0]
        assert num_points_gss >= (num_samples*3)
        anchors = farthest_point_sampling_fix_first(points_gss, num_samples)
    # Note that the sampling result generatd by grid-subsampling is NOT a subset of the original input point set!
    return anchors


def visualize_para_points(para_points, scaling_factor=1):
    para_points = para_points.detach().cpu()
    assert para_points.ndim==3 and para_points.size(2)==3
    batch_size, num_pixels = para_points.size(0), para_points.size(1)
    is_square_number(num_pixels)
    img_res = int(np.sqrt(num_pixels))
    para_img = np.asarray(para_points.permute(0, 2, 1).contiguous().view(batch_size, -1, img_res, img_res))
    img_list = []
    for bid in range(batch_size):
        img = show_image(para_img[bid])
        resized_img_res = int(np.around(img_res * scaling_factor))
        img = img.resize((resized_img_res, resized_img_res), Image.BILINEAR)
        img_list.append(img)
    return img_list


class SMLP(nn.Module):
    def __init__(self, ic, oc, is_bn, nl, slope=None):
        super(SMLP, self).__init__()
        assert isinstance(is_bn, bool)
        assert nl in ['none', 'relu', 'leakyrelu', 'tanh', 'sigmoid']
        if nl == 'leakyrelu':
            assert slope is not None
        if slope is not None:
            assert nl == 'leakyrelu'
            assert slope>=0 and slope<=1
        self.is_bn = is_bn
        self.nl = nl
        self.conv = nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=1, bias=False)
        if self.is_bn:
            self.bn = nn.BatchNorm2d(oc)
        if nl == 'relu':
            self.activate = nn.ReLU(inplace=True)
        if nl == 'leakyrelu':
            self.activate = nn.LeakyReLU(negative_slope=slope, inplace=True)
        if nl == 'tanh':
            self.activate = nn.Tanh()
        if nl == 'sigmoid':
            self.activate = nn.Sigmoid()
    def forward(self, x):
        # x: [B, N, ic]
        # y: [B, N, oc]
        x = x.permute(0, 2, 1).contiguous().unsqueeze(-1) # [B, ic, N, 1]
        y = self.conv(x) # [B, oc, N, 1]
        if self.is_bn:
            y = self.bn(y)
        if self.nl != 'none':
            y = self.activate(y)   
        y = y.squeeze(-1).permute(0, 2, 1).contiguous() # [B, N, oc]
        return y
    
    
class CU(nn.Module):
    def __init__(self, ic, oc, ks, is_bn, nl, slope=None, pad='zeros'):
        super(CU, self).__init__()
        assert np.mod(ks + 1, 2) == 0
        assert isinstance(is_bn, bool)
        assert nl in ['none', 'relu', 'leakyrelu', 'tanh', 'sigmoid']
        if nl == 'leakyrelu':
            assert slope is not None
        if slope is not None:
            assert nl == 'leakyrelu'
            assert slope>=0 and slope<=1
        assert pad in ['zeros', 'reflect', 'replicate', 'circular']
        self.is_bn = is_bn
        self.nl = nl
        self.conv = nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=ks, stride=1, 
                    padding=(ks-1)//2, dilation=1, groups=1, bias=False, padding_mode=pad)
        if self.is_bn:
            self.bn = nn.BatchNorm2d(oc)
        if nl == 'relu':
            self.activate = nn.ReLU(inplace=True)
        if nl == 'leakyrelu':
            self.activate = nn.LeakyReLU(negative_slope=slope, inplace=True)
        if nl == 'tanh':
            self.activate = nn.Tanh()
        if nl == 'sigmoid':
            self.activate = nn.Sigmoid()
    def forward(self, x):
        # x: [B, ic, H, W]
        # y: [B, oc, H, W]
        y = self.conv(x) # [B, oc, H, W]
        if self.is_bn:
            y = self.bn(y)
        if self.nl != 'none':
            y = self.activate(y)
        return y


class FC(nn.Module):
    def __init__(self, ic, oc, is_bn, nl, slope=None):
        super(FC, self).__init__()
        assert isinstance(is_bn, bool)
        assert nl in ['none', 'relu', 'leakyrelu', 'tanh', 'sigmoid']
        if nl == 'leakyrelu':
            assert slope is not None
        if slope is not None:
            assert nl == 'leakyrelu'
            assert slope>=0 and slope<=1
        self.is_bn = is_bn
        self.nl = nl
        self.linear = nn.Linear(ic, oc, bias=False)
        if self.is_bn:
            self.bn = nn.BatchNorm1d(oc)
        if nl == 'relu':
            self.activate = nn.ReLU(inplace=True)
        if nl == 'leakyrelu':
            self.activate = nn.LeakyReLU(negative_slope=slope, inplace=True)
        if nl == 'tanh':
            self.activate = nn.Tanh()
        if nl == 'sigmoid':
            self.activate = nn.Sigmoid()
    def forward(self, x):
        # x: [B, ic]
        # y: [B, oc]
        y = self.linear(x) # [B, oc]
        if self.is_bn:
            y = self.bn(y)
        if self.nl != 'none':
            y = self.activate(y)
        return y


class ResConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, leaky_slope):
        super(ResConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False, padding_mode='replicate'), 
            nn.BatchNorm2d(in_channels)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False, padding_mode='replicate'), 
            nn.BatchNorm2d(out_channels)
        )
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False), nn.BatchNorm2d(out_channels))
        self.nl = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
    def forward(self, in_ftr):
        out_ftr = self.conv_2(self.nl(self.conv_1(in_ftr)))
        if self.in_channels != self.out_channels:
            out_ftr = self.nl(self.shortcut(in_ftr) + out_ftr)
        else:
            out_ftr = self.nl(in_ftr + out_ftr)
        return out_ftr
    
    
class DecConv2D(nn.Module):
    def __init__(self, in_channels, leaky_slope):
        super(DecConv2D, self).__init__()
        dec_1 = CU(in_channels, 32, 1, True, 'leakyrelu', leaky_slope)
        dec_2 = CU(32, 16, 1, True, 'leakyrelu', leaky_slope)
        dec_3 = CU(16, 3, 1, False, 'none')
        self.dec = nn.Sequential(dec_1, dec_2, dec_3)
    def forward(self, x):
        y = self.dec(x)
        return y
    
    
class ResSMLP(nn.Module):
    def __init__(self, in_channels, out_channels, leaky_slope):
        super(ResSMLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.smlp_1 = SMLP(in_channels, in_channels, True, 'none')
        self.smlp_2 = SMLP(in_channels, out_channels, True, 'none')
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


class GlobalAnchorEmbedding(nn.Module):
    def __init__(self, lat_res, leaky_slope=0.0):
        super(GlobalAnchorEmbedding, self).__init__()
        self.lat_res = lat_res
        self.num_grids = lat_res ** 2
        grid_points_raw, _ = build_lattice(lat_res, lat_res)
        self.grid_points = torch.tensor(grid_points_raw).permute(1, 0).contiguous().view(2, lat_res, lat_res)
        self.enc_0 = CU(2, 16, 1, True, 'leakyrelu', leaky_slope)
        self.enc_1 = ResConv2D(16, 64, leaky_slope)
        self.enc_2 = ResConv2D(64, 128, leaky_slope)
        self.enc_3 = ResConv2D(128, 256, leaky_slope)
        self.dec_1 = DecConv2D(64, leaky_slope)
        self.dec_2 = DecConv2D(128, leaky_slope)
        self.dec_3 = DecConv2D(256, leaky_slope)
    def upscale(self, in_features):
        out_features = F.interpolate(in_features, scale_factor=2, mode='bilinear')
        return out_features
    def forward(self, anc_pts):
        B = anc_pts.size(0)
        M = anc_pts.size(1)
        assert anc_pts.ndim==3 and anc_pts.size(2)==3
        device = anc_pts.device
        lat = self.grid_points.to(device).unsqueeze(0).repeat(B, 1, 1, 1)
        ftr_0 = self.enc_0(lat)
        ftr_1 = self.enc_1(ftr_0)
        ftr_2 = self.enc_2(self.upscale(ftr_1))
        ftr_3 = self.enc_3(self.upscale(ftr_2))
        img_1 = self.dec_1(ftr_1)
        img_2 = self.dec_2(ftr_2)
        img_3 = self.dec_3(ftr_3)
        pts_1 = img_1.view(B, 3, -1).permute(0, 2, 1).contiguous()
        pts_2 = img_2.view(B, 3, -1).permute(0, 2, 1).contiguous()
        pts_3 = img_3.view(B, 3, -1).permute(0, 2, 1).contiguous()
        return pts_1, pts_2, pts_3


class NeighborsAggregation(nn.Module):
    def __init__(self, in_channels, K, leaky_slope):
        super(NeighborsAggregation, self).__init__()
        self.K = K
        self.smlp = SMLP(in_channels*2, in_channels, True, 'leakyrelu', leaky_slope)
        self.ap = AttPool(in_channels)
    def forward(self, pts, pwf):
        assert pts.size(0)==pwf.size(0) and pts.size(1)==pwf.size(1) and pts.size(2)==3
        B, N, C = pwf.size()
        pwf_d = pwf.unsqueeze(2).repeat(1, 1, self.K, 1)
        knn_idx = knn_search(pts, pts, self.K+1)[:, :, 1:]
        pwf_n = index_points(pwf, knn_idx)
        pwf_e = torch.cat((pwf_d, pwf_n-pwf_d), dim=-1)
        pwf_agg = self.ap(self.smlp(pwf_e.view(B*N, self.K, -1))).view(B, N, -1)
        return pwf_agg
    
    
class PatchEncoder(nn.Module):
    def __init__(self, out_channels, K, leaky_slope):
        super(PatchEncoder, self).__init__()
        self.out_channels = out_channels
        self.K = K
        self.layer_1 = nn.Sequential(SMLP(3, 12, True, 'leakyrelu', leaky_slope), ResSMLP(12, 20, leaky_slope))
        self.layer_2 = NeighborsAggregation(20, K, leaky_slope)
        self.layer_3 = ResSMLP(40, out_channels, leaky_slope)
    def forward(self, pts):
        assert pts.ndim==3 and pts.size(2)==3
        B = pts.size(0)
        N = pts.size(1)
        ftr_1 = self.layer_1(pts)
        ftr_2 = self.layer_2(pts, ftr_1)
        ftr_2_fused = torch.cat((ftr_2, ftr_2.max(dim=1, keepdim=True)[0].repeat(1, N, 1)), dim=-1)
        pwf = self.layer_3(ftr_2_fused)
        cdw = pwf.mean(dim=1)
        return pwf, cdw


class LocalPatchEmbedding(nn.Module):
    def __init__(self, Ke=4, Kc=4, compute_mec=False, leaky_slope=0.01):
        super(LocalPatchEmbedding, self).__init__()
        self.Ke = Ke
        self.Kc = Kc
        self.compute_mec = compute_mec
        assert Ke == Kc
        self.encoder = PatchEncoder(128, Ke, leaky_slope)
        unf_1 = SMLP(131, 64, True, 'leakyrelu', leaky_slope)
        unf_2 = SMLP(64, 32, True, 'leakyrelu', leaky_slope)
        unf_3 = SMLP(32, 2, False, 'none')
        self.unfold = nn.Sequential(unf_1, unf_2, unf_3, nn.Sigmoid())
        reg_1 = SMLP(128, 128, True, 'leakyrelu', leaky_slope)
        reg_2 = SMLP(128, 64, True, 'leakyrelu', leaky_slope)
        reg_3 = SMLP(64, Kc, False, 'none')
        self.regress = nn.Sequential(reg_1, reg_2, reg_3, nn.Softmax(dim=-1))
    def forward(self, pts):
        assert pts.ndim==3 and pts.size(2)==3
        B = pts.size(0)
        N = pts.size(1)
        pwf, cdw = self.encoder(pts)
        pe = self.unfold(torch.cat((pts, cdw.unsqueeze(1).repeat(1, N, 1)), dim=-1))
        if self.compute_mec:
            lcw = self.regress(pwf)
            mec_p, mec_e = manifold_embedding_constraint(pts, pe, lcw)
            return pe, lcw, mec_p, mec_e
        else:
            return pe


class BoundaryConnectivityAlignment(nn.Module):
    def __init__(self, leaky_slope=0.0):
        super(BoundaryConnectivityAlignment, self).__init__()
        self.conv_1 = nn.Sequential(CU(3, 32, 3, True, 'leakyrelu', leaky_slope), ResConv2D(32, 32, leaky_slope))
        self.conv_2 = ResConv2D(32, 64, leaky_slope)
        self.conv_3 = ResConv2D(64, 128, leaky_slope)
        self.conv_4 = ResConv2D(256, 256, leaky_slope)
        self.conv_5 = ResConv2D(256, 128, leaky_slope)
        self.output = nn.Sequential(CU(256, 128, 1, True, 'leakyrelu', leaky_slope),CU(128, 1, 1, False, 'sigmoid'))
    def forward(self, sep_pat_para, in_deepgi):
        assert sep_pat_para.size(0) == in_deepgi.size(0)
        assert sep_pat_para.size(2)==3 and sep_pat_para.size(3)==sep_pat_para.size(4)
        assert in_deepgi.size(1)==3 and in_deepgi.size(2)==in_deepgi.size(3)
        B = sep_pat_para.size(0)
        M = sep_pat_para.size(1)
        r = sep_pat_para.size(3)
        s = in_deepgi.size(2)
        is_square_number(M)
        m = int(np.sqrt(M))
        ftr_1 = self.conv_1(in_deepgi)
        ftr_2 = self.conv_2(F.max_pool2d(ftr_1, 2))
        ftr_3 = self.conv_3(F.max_pool2d(ftr_2, 2))
        ftr_3_fused = torch.cat((ftr_3, F.adaptive_avg_pool2d(ftr_3, 1).repeat(1, 1, s//4, s//4)), dim=1)
        ftr_4 = self.conv_4(ftr_3_fused)
        ftr_4_pooled = F.adaptive_avg_pool2d(ftr_4, (m, m))
        ftr_5 = self.conv_5(ftr_4_pooled)
        ftr_5_fused = torch.cat((ftr_5, F.adaptive_avg_pool2d(ftr_5, 1).repeat(1, 1, m, m)), dim=1)
        rot_angles = self.output(ftr_5_fused) * 360.0
        rot_angles = rot_angles.squeeze(1).view(B, M)
        sep_pat_para_rot = []
        for bid in range(B):
            sep_pat_para_rot.append(rotate_pat_para(sep_pat_para[bid, ...], rot_angles[bid, :]).unsqueeze(0))
        sep_pat_para_rot = torch.cat(sep_pat_para_rot, dim=0)
        out_deepgi = assemble_separate_patch_parameterizations(sep_pat_para_rot)
        return out_deepgi, rot_angles
    
    
def normalize_anchor_patches(anc_pat):
    B = anc_pat.size(0)
    M = anc_pat.size(1)
    K = anc_pat.size(2)
    C = anc_pat.size(3)
    assert anc_pat.ndim==4 and C>=3
    coor = anc_pat[:, :, :, 0:3]
    if C > 3:
        atr = anc_pat[:, :, :, 3:].view(B, M, K, -1)
    coor_nrm = coor.view(B*M, K, 3)
    centroids = torch.mean(coor_nrm, dim=1, keepdim=True)
    coor_nrm = coor_nrm - centroids
    distances = torch.sqrt(torch.abs(torch.sum(coor_nrm**2, dim=-1, keepdim=True)))
    max_distances = torch.max(distances, dim=1, keepdim=True)[0]
    coor_nrm = coor_nrm / max_distances
    coor_nrm = coor_nrm.view(B, M, K, 3)
    if C == 3:
        anc_pat_nrm = coor_nrm
    else:
        anc_pat_nrm = torch.cat((coor_nrm, atr), dim=-1)
    return anc_pat_nrm


class AttPool(nn.Module):
    def __init__(self, in_chs):
        super(AttPool, self).__init__()
        self.in_chs = in_chs
        self.linear_transform = SMLP(in_chs, in_chs, False, 'none')
    def forward(self, x):
        bs = x.size(0)
        num_pts = x.size(1)
        assert x.ndim==3 and x.size(2)==self.in_chs
        scores = F.softmax(self.linear_transform(x), dim=1)
        y = (x * scores).sum(dim=1)
        return y
    
def manifold_embedding_constraint(P, E, W):
    B, N, K = W.size()
    assert P.size(0)==B and P.size(1)==N
    assert E.size(0)==B and E.size(1)==N
    assert P.size(2)==3 and E.size(2)==2
    knn_idx = knn_search(P, P, K+1)[:, :, 1:]
    Pn = index_points(P, knn_idx)
    En = index_points(E, knn_idx)
    Pr = torch.sum(W.unsqueeze(-1) * Pn, dim=2)
    Er = torch.sum(W.unsqueeze(-1) * En, dim=2)
    rec_err_on_P = torch.sum((P - Pr) ** 2, dim=-1)
    rec_err_on_E = torch.sum((E - Er) ** 2, dim=-1)
    mec_p = rec_err_on_P.mean()
    mec_e = rec_err_on_E.mean()
    return mec_p, mec_e


def draw_pe(pe, marker_size, save_path=None):
    assert pe.ndim==2 and pe.size(1)==2
    pe = np.asarray(pe.detach().cpu())
    fig = plt.figure(figsize=(4, 4))
    plt.xlim(-0.02, +1.02)
    plt.ylim(-0.02, +1.02)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(alpha=0.5, linestyle='--', linewidth=1)
    marker_color = np.asarray([120, 150, 200]).reshape(1, -1) / 255.0
    x_for_draw = pe[:, 1]
    y_for_draw = 1.0 - pe[:, 0]
    plt.scatter(x_for_draw, y_for_draw, s=marker_size, marker='.', c=marker_color)
    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=200)
        plt.close()


def draw_pe_cc(pts, pe, marker_size, save_path=None):
    assert pts.ndim==2 and pe.ndim==2
    assert pts.size(0) == pe.size(0)
    assert pts.size(1)==3 and pe.size(1)==2
    pts = np.asarray(pts.detach().cpu())
    pe = np.asarray(pe.detach().cpu())
    fig = plt.figure(figsize=(4, 4))
    plt.xlim(-0.02, +1.02)
    plt.ylim(-0.02, +1.02)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(alpha=0.5, linestyle='--', linewidth=1)
    cc = min_max_normalization(centroid_normalization(pts))
    x_for_draw = pe[:, 1]
    y_for_draw = 1.0 - pe[:, 0]
    plt.scatter(x_for_draw, y_for_draw, s=marker_size, marker='.', c=cc)
    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=200)
        plt.close()
        
        
def rescale_pe(pe, range_min, range_max):
    assert pe.ndim==3 and pe.size(2)==2
    assert range_min < range_max
    values_min = torch.min(pe, dim=1, keepdim=True)[0]
    values_max = torch.max(pe, dim=1, keepdim=True)[0]
    pe_rescaled = (pe - values_min) / (values_max - values_min)
    pe_rescaled = pe_rescaled * (range_max - range_min) + range_min
    return pe_rescaled


def repulsion_function(x, t_min):
    y = F.relu(-x + t_min)
    return y


def repulsion_loss(pe, min_nnd):
    assert pe.size(2) == 2
    B = pe.size(0)
    N = pe.size(1)
    nn_idx = knn_search(pe, pe, 2)[:, :, 1]
    pe_nn = index_points(pe, nn_idx)
    nnd_squared = torch.sum((pe-pe_nn)**2, dim=-1)
    min_nnd_squared = min_nnd ** 2
    err = repulsion_function(nnd_squared, min_nnd_squared)
    rep_loss = err.mean()
    return rep_loss


def seperately_grid_resample_patches(pat_pts, pe, R):
    is_square_number(R)
    r = int(np.sqrt(R))
    assert pat_pts.size(0) == pe.size(0)
    assert pat_pts.size(1) == pe.size(1)
    assert pat_pts.size(2) == pe.size(2)
    assert pat_pts.size(3)==3 and pe.size(3)==2
    B = pat_pts.size(0)
    M = pat_pts.size(1)
    K = pat_pts.size(2)
    sep_pat_para = grid_resample(pat_pts.view(B*M, K, 3), pe.view(B*M, K, 2), R)
    sep_pat_para = sep_pat_para.permute(0, 2, 1).contiguous().view(B*M, 3, r, r).view(B, M, 3, r, r)
    return sep_pat_para


def assemble_separate_patch_parameterizations(sep_pat_para):
    assert sep_pat_para.size(2) == 3
    assert sep_pat_para.size(3) == sep_pat_para.size(4)
    B = sep_pat_para.size(0)
    M = sep_pat_para.size(1)
    r = sep_pat_para.size(3)
    device = sep_pat_para.device
    is_square_number(M)
    m = int(np.sqrt(M))
    s = m * r
    raw_deepgi = torch.empty(B, 3, s, s).to(device)
    for i in range(m):
        for j in range(m):
            raw_deepgi[:, :, i*r:(i+1)*r, j*r:(j+1)*r] = sep_pat_para[:, i*m+j, ...]
    return raw_deepgi


def rotate_pat_para(pat_para, angles):
    assert pat_para.size(0) == angles.size(0)
    height, width = pat_para.size(2), pat_para.size(3)
    pad_h, pad_w = int(np.ceil(height*0.2)), int(np.ceil(width*0.2))
    pat_para_pad = F.interpolate(pat_para, size=(height+2*pad_h, width+2*pad_w), mode='nearest')
    pat_para_pad_rot = kornia.geometry.transform.rotate(pat_para_pad, angles, align_corners=True)
    pat_para_pad_rot_crp = pat_para_pad_rot[:, :, pad_h:(height+pad_h), pad_w:(width+pad_w)]
    return pat_para_pad_rot_crp


def block_edge_connectivity(deepgi, m, r):
    assert deepgi.size(1)==3 and deepgi.size(2)==deepgi.size(3)
    B = deepgi.size(0)
    s = deepgi.size(2)
    assert s == (m*r)
    edge_points_1 = []
    edge_points_2 = []
    for cid in range(1, m):
        c1 = deepgi[:, :, :, (cid*r)-1].unsqueeze(1)
        c2 = deepgi[:, :, :, cid*r].unsqueeze(1)
        edge_points_1.append(c1)
        edge_points_2.append(c2)
    for rid in range(1, m):
        r1 = deepgi[:, :, (rid*r)-1, :].unsqueeze(1)
        r2 = deepgi[:, :, rid*r, :].unsqueeze(1)
        edge_points_1.append(r1)
        edge_points_2.append(r2)
    edge_points_1 = torch.cat(edge_points_1, dim=1).permute(0, 1, 3, 2).contiguous()
    edge_points_2 = torch.cat(edge_points_2, dim=1).permute(0, 1, 3, 2).contiguous()
    err = torch.sum(((edge_points_1-edge_points_2)**2), dim=-1)
    loss = err.mean()
    return loss


def total_variation(img):
    grads = kornia.filters.SpatialGradient()(img)
    tv_loss = grads.abs().mean()
    return tv_loss


def rotate_pe(pe, rot_angles):
    assert pe.size(0)==rot_angles.size(0) and pe.size(1)==rot_angles.size(1)
    assert pe.size(3) == 2
    K = pe.size(2)
    center_u, center_v = 0.5, 0.5
    rot_angles_expand = rot_angles.unsqueeze(-1).repeat(1, 1, K) * np.pi / 180.0
    rot_angles_cos = torch.cos(rot_angles_expand)
    rot_angles_sin = torch.sin(rot_angles_expand)
    pe_u = pe[:, :, :, 0]
    pe_v = pe[:, :, :, 1]
    pe_u_rot = (pe_u - center_u) * rot_angles_cos - (pe_v - center_v) * rot_angles_sin + center_u
    pe_v_rot = (pe_u - center_u) * rot_angles_sin + (pe_v - center_v) * rot_angles_cos + center_v
    pe_rotated = torch.cat((pe_u_rot.unsqueeze(-1), pe_v_rot.unsqueeze(-1)), dim=-1)
    return pe_rotated


def grid_resample(pts, pe, num_rsp):
    assert pts.ndim==3 and pe.ndim==3
    assert pts.size(0)==pe.size(0) and pts.size(1)==pe.size(1)
    assert pts.size(2)==3 and pe.size(2)==2
    is_square_number(num_rsp)
    lat_res = int(np.sqrt(num_rsp))
    batch_size, num_points, device = pe.size(0), pe.size(1), pe.device
    grid_points, _ = build_lattice_01(lat_res, lat_res)
    grid_points = torch.tensor(grid_points).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    nn_idx = knn_search(pe.detach(), grid_points.detach(), 1).squeeze(-1)
    pts_rsp = index_points(pts, nn_idx)
    return pts_rsp


