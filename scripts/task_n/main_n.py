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


train_bs = 16
train_set = MN40RegParaLoader(data_root, 'train')
train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True, num_workers=8, worker_init_fn=seed_worker, drop_last=True)


net = RegGeoNetReg().cuda()
max_lr = 5e-2
min_lr = 5e-5
num_epc = 200
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=max_lr, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epc, eta_min=min_lr)


m = 512
M = (m ** 2)
for epc in range(1, num_epc+1):
    net.train()
    epoch_loss = 0
    num_samples = 0
    for (gi_points, gi_normals_gt) in tqdm(train_loader):
        optimizer.zero_grad()
        gi_points = gi_points.cuda()
        gi_normals_gt = gi_normals_gt.cuda()
        bs = gi_points.size(0)
        gi_normals_pr = net(gi_points)
        gi_normals_pr_reshaped = gi_normals_pr.view(bs, 3, -1).permute(0, 2, 1).contiguous()
        gi_normals_gt_reshaped = gi_normals_gt.view(bs, 3, -1).permute(0, 2, 1).contiguous()
        ndists = compute_normals_distances(gi_normals_pr_reshaped, gi_normals_gt_reshaped)
        loss = ndists.mean()
        loss.backward()
        optimizer.step()
        num_samples += bs
        epoch_loss += (loss.item() * bs)
    scheduler.step()
    epoch_loss = np.around(epoch_loss/num_samples, 4)
    torch.save({'model': net.state_dict(), 'optim': optimizer.state_dict()}, ckpt_path)
    print('epoch: {}, loss: {}'.format(epc, epoch_loss))
    
    
    