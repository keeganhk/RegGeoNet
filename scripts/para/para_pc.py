import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
prj_root = os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(prj_root)
from utils.pkgs import *
from utils.generic_cpnts import *
from utils.para_cpnts import *


load_path = 'xxx.npy'
pc_xyz = bounding_box_normalization(np.load(load_path))[:, 0:3] # (N, 3)
pts = torch.tensor(pc_xyz).unsqueeze(0).float().cuda() # [B, N, 3]
B, N, _ = pts.size()


N_A = 1024 # number of global anchor points; each anchor corresponds to a local patch
K_L = 196 # number of points contained in each local patch
N_L = 256 # number of pixels resampled for each local patch parameterization
n_a = int(np.sqrt(N_A)) # the resolution of the global anchor parameterization is "n_a x n_a"
n_l = int(np.sqrt(N_L)) # the resolution of each local patch parameterization is "n_l x n_l"
M = N_A * N_L # number of pixels contained in the resulting DeepGI
m = n_a * n_l # the resolution of the resulting DeepGI is "m x m"
assert M == m**2
is_square_number(N_A)
is_square_number(N_L)
# for convenience, we configure both N_A and N_L as square numbers to produce a square image;
# however, in practice users can flexibly specify N_A and N_L to produce a DeepGI with arbitrary aspect ratio
n_a_2 = n_a
n_a_1 = n_a // 2
n_a_0 = n_a // 4
N_A_2 = n_a_2 ** 2
N_A_1 = n_a_1 ** 2
N_A_0 = n_a_0 ** 2


# torch.set_default_dtype(torch.float64)
# seed = 0
# random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

gae_module = GlobalAnchorEmbedding(n_a_0).cuda()
gae_module.load_state_dict(torch.load('../../ckpt/para_ckpt/gae_initialized.pth'))
gae_module.train()
gae_max_lr = 1e-3
gae_min_lr = 1e-5
gae_num_epc = 300
gae_optimizer = optim.AdamW(gae_module.parameters(), lr=gae_max_lr, weight_decay=1e-8)
gae_scheduler = optim.lr_scheduler.CosineAnnealingLR(gae_optimizer, T_max=gae_num_epc, eta_min=gae_min_lr)

lpe_module = LocalPatchEmbedding().cuda()
lpe_module.load_state_dict(torch.load('../../ckpt/para_ckpt/lpe_pretrained.pth'))
lpe_module.eval()

bca_module = BoundaryConnectivityAlignment().cuda()
bca_module.load_state_dict(torch.load('../../ckpt/para_ckpt/bca_initialized.pth'))
bca_module.train()
bca_max_lr = 1e-4
bca_min_lr = 1e-4
bca_num_epc = 10
bca_optimizer = optim.AdamW(bca_module.parameters(), lr=bca_max_lr, weight_decay=1e-8)
bca_scheduler = optim.lr_scheduler.CosineAnnealingLR(bca_optimizer, T_max=bca_num_epc, eta_min=bca_min_lr)


anc_points_2 = index_points(pts, fps_fix_first(pts, N_A))
anc_points_1 = index_points(anc_points_2, fps_fix_first(anc_points_2, N_A_1))
anc_points_0 = index_points(anc_points_1, fps_fix_first(anc_points_1, N_A_0))
# anc_points_2 = torch.tensor(anchor_sampling(pc_xyz, N_A, False)).unsqueeze(0).cuda()
# anc_points_1 = index_points(anc_points_2, fps_fix_first(anc_points_2, N_A_1))
# anc_points_0 = index_points(anc_points_1, fps_fix_first(anc_points_1, N_A_0))
for gae_epc in range(1, gae_num_epc+1):
    gae_optimizer.zero_grad()
    gae_outputs_0, gae_outputs_1, gae_outputs_2 = gae_module(anc_points_2)
    gae_loss_1 = earth_mover_distance_cuda(anc_points_0, gae_outputs_0)
    gae_loss_2 = earth_mover_distance_cuda(anc_points_1, gae_outputs_1)
    gae_loss_3 = earth_mover_distance_cuda(anc_points_2, gae_outputs_2)
    gae_loss = gae_loss_1 + gae_loss_2 + gae_loss_3
    gae_loss.backward()
    gae_optimizer.step()
    gae_scheduler.step()
para_a = index_points(anc_points_2, knn_search(anc_points_2.detach().cpu(), gae_outputs_2.detach().cpu(), 1).squeeze(-1))

pat_pts = index_points(pts, knn_search(pts.cpu(), para_a.cpu(), K_L))
pat_pts_n = normalize_anchor_patches(pat_pts)
with torch.no_grad():
    lpe_inputs = pat_pts_n.view(B*N_A, K_L, 3)
    pe = lpe_module(lpe_inputs)
    pe = rescale_pe(pe, 0+1e-6, 1-1e-6).view(B, N_A, K_L, 2)
sep_pat_para = seperately_grid_resample_patches(pat_pts, pe, N_L)
raw_deepgi = assemble_separate_patch_parameterizations(sep_pat_para)

edge_only = True
for bca_epc in range(1, bca_num_epc+1):
    out_deepgi, rot_angles = bca_module(sep_pat_para, raw_deepgi)
    bca_optimizer.zero_grad()
    if edge_only:
        bca_loss = block_edge_connectivity(out_deepgi, n_a, n_l)
    else:
        bca_loss = total_variation(out_deepgi)
    bca_loss.backward()
    bca_optimizer.step()
    bca_scheduler.step()
refined_pe = rescale_pe(rotate_pe(pe, rot_angles).view(B*N_A, K_L, 2), 0+1e-6, 1-1e-6).view(B, N_A, K_L, 2)
refined_sep_pat_para = seperately_grid_resample_patches(pat_pts, refined_pe, N_L)
deepgi = assemble_separate_patch_parameterizations(refined_sep_pat_para)

deepgi_as_img = show_image(deepgi.squeeze(0).cpu())
deepgi_as_pts = deepgi.view(B, 3, -1).permute(0, 2, 1).contiguous()


