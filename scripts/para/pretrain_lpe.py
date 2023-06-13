import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
prj_root = os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(prj_root)
from utils.pkgs import *
from utils.generic_cpnts import *
from utils.para_cpnts import *


data_root = '../../data/LocalPatches'
ckpt_root = '../../ckpt/para_ckpt'
ckpt_path = os.path.join(ckpt_root, 'lpe_pretrained.pth')
K = 128
data_file = os.path.join(data_root, 'patch_points_' + align_number(K, 3) + '.h5')


class PatchLoader(torch.utils.data.Dataset):
    def __init__(self, data_file):
        self.data_file = data_file
        fid = h5py.File(data_file, 'r')
        self.num_patches = fid['data'].shape[0]
        fid.close()
    def __getitem__(self, index):
        fid = h5py.File(self.data_file, 'r')
        pat_pts_n = bounding_box_normalization(fid['data'][index]).astype(np.float32)
        fid.close()        
        pat_pts_n = random_rotation(pat_pts_n)
        return pat_pts_n
    def __len__(self):
        return self.num_patches


tr_bs = 256
tr_set = PatchLoader(data_file)
tr_loader = DataLoader(tr_set, batch_size=tr_bs, shuffle=True, num_workers=12, drop_last=True)


lpe_module = LocalPatchEmbedding(4, 4, True, 0.01).cuda()
init_lr = 1e-1
min_lr = 1e-4
train_epoch = 120
optimizer = optim.AdamW(lpe_module.parameters(), lr=init_lr, weight_decay=1e-8)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epoch, eta_min=min_lr)
rep_th = 1 / (np.sqrt(K) - 1) * 0.5
gamma_list = list(np.linspace(1, 0, int(train_epoch*0.25))) + list(np.zeros((int(train_epoch*0.75))))


lpe_module.train()
for epoch_index in range(1, train_epoch+1):
    num_fed = 0
    epoch_rep_loss = 0
    epoch_mec_loss = 0
    for pat_pts_n in tqdm(tr_loader):
        pat_pts_n = pat_pts_n.cuda()
        B = pat_pts_n.size(0)
        optimizer.zero_grad()
        pe, lcw, mec_p, mec_e = lpe_module(pat_pts_n)
        loss_rep = repulsion_loss(pe, rep_th)
        loss_mec = (mec_p + mec_e) / 2
        loss = loss_rep + loss_mec * gamma_list[epoch_index-1]
        num_fed += B
        epoch_rep_loss += (loss_rep.item() * B)
        epoch_mec_loss += (loss_mec.item() * B)
        loss.backward()
        optimizer.step()
    scheduler.step()
    epoch_rep_loss = np.around(epoch_rep_loss/num_fed, 8)
    epoch_mec_loss = np.around(epoch_mec_loss/num_fed, 8)
    print('epoch: {}, rep_loss: {}, mec_loss: {}'.format(epoch_index, epoch_rep_loss, epoch_mec_loss))
    torch.save(lpe_module.state_dict(),  ckpt_path)
    
    
    