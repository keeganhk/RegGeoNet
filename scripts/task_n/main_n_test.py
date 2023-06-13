import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
prj_root = os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(prj_root)
from utils.pkgs import *
from utils.generic_cpnts import *
from utils.para_cpnts import *


from task_n_utils import *
data_root = '../../data/ModelNet40'
ckpt_root = '../../ckpt/task_ckpt'
ckpt_path = os.path.join(ckpt_root, 'reggeonet_reg_mn40.pth')


net = RegGeoNetReg().cuda()
net.load_state_dict(torch.load(ckpt_path)['model'])
net.eval()

test_bs = 16
test_set = MN40RegParaLoader(data_root, 'test')
test_loader = DataLoader(test_set, batch_size=test_bs, shuffle=False, num_workers=8, worker_init_fn=seed_worker, drop_last=False)

ndists_mean_collection = []
for (gi_points, gi_normals_gt) in tqdm(test_loader):
    gi_points = gi_points.cuda()
    gi_normals_gt = gi_normals_gt.cuda()
    bs = gi_points.size(0)
    with torch.no_grad():
        gi_normals_pr = net(gi_points)
    gi_normals_pr_reshaped = gi_normals_pr.view(bs, 3, -1).permute(0, 2, 1).contiguous()
    gi_normals_gt_reshaped = gi_normals_gt.view(bs, 3, -1).permute(0, 2, 1).contiguous()
    ndists_mean = compute_normals_distances(gi_normals_pr_reshaped, gi_normals_gt_reshaped).mean(dim=-1)
    ndists_mean_collection.append(ndists_mean)
ndists_mean_collection = np.asarray(torch.cat(ndists_mean_collection).cpu())
print('mean test error: {}'.format(np.around(float(ndists_mean_collection.mean()), 2)))


