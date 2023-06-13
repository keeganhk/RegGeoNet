import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
prj_root = os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(prj_root)
from utils.pkgs import *
from utils.generic_cpnts import *
from utils.para_cpnts import *


from task_c_sparse_utils import *
data_root = '../../data/ScanObjectNN'
ckpt_root = '../../ckpt/task_ckpt'
# ckpt_path = os.path.join(ckpt_root, 'reggeonet_cls_sonn_bg.pth')
ckpt_path = os.path.join(ckpt_root, 'reggeonet_cls_sonn_only.pth')


net = RegGeoNetClsSparse(num_classes=15).cuda()
net.load_state_dict(torch.load(ckpt_path)['model'])
net.eval()

test_bs = 64
# test_set = SONNBGClsParaLoader(data_root, 'test')
test_set = SONNONLYClsParaLoader(data_root, 'test')
test_loader = DataLoader(test_set, batch_size=test_bs, shuffle=False, num_workers=8, worker_init_fn=seed_worker, drop_last=False)

num_samples = 0
num_correct = 0
for (gi_img, cid) in tqdm(test_loader):
    gi_img = gi_img.cuda()
    cid = cid.long().cuda()
    bs = gi_img.size(0)
    with torch.no_grad():
        logits = net(gi_img)
    preds = logits.argmax(dim=-1).detach()
    num_samples += bs
    num_correct += (preds==cid).sum().item()
test_acc = np.around((num_correct/num_samples)*100, 2)
print('best test acc: {}%'.format(test_acc))


