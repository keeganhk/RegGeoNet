import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
prj_root = os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(prj_root)
from utils.pkgs import *
from utils.generic_cpnts import *
from utils.para_cpnts import *


from task_s_utils import *
data_root = '../../data/ShapeNetPart'
ckpt_root = '../../ckpt/task_ckpt'
ckpt_path = os.path.join(ckpt_root, 'reggeonet_seg_snp.pth')


net = RegGeoNetSeg(num_object_classes=16, num_part_classes=50).cuda()
net.load_state_dict(torch.load(ckpt_path))
net.eval()

snp_objects_names, snp_objects_parts = ShapeNetPart_ObjectsParts()
test_bs = 12
test_set = SNPSegParaLoader(data_root, 'test')
test_loader = DataLoader(test_set, batch_size=test_bs, shuffle=False, num_workers=8, worker_init_fn=seed_worker, drop_last=False)


m = 512
M = (m ** 2)
iou_list = []
for (G, L_gt, cid, name_list) in tqdm(test_loader):
    G = G.cuda()
    L_gt = L_gt.long().cuda()
    cid = cid.long().cuda()
    bs = G.size(0)
    with torch.no_grad():
        logits = net(G, cid)
        logits_reshaped = logits.view(bs, -1, M).permute(0, 2, 1).contiguous()
    preds = np.asarray(logits_reshaped.argmax(dim=-1).cpu())
    labels = np.asarray(L_gt.squeeze(1).view(bs, M).cpu())
    for bid in range(bs):
        L_this = labels[bid]
        P_this = preds[bid]
        class_name = name_list[bid][:-5]
        parts = snp_objects_parts[class_name]
        this_parts_iou = []
        for part_this in parts:
            if (L_this==part_this).sum() == 0:
                this_parts_iou.append(1.0)
            else:
                I = np.sum(np.logical_and(P_this==part_this, L_this==part_this))
                U = np.sum(np.logical_or(P_this==part_this, L_this==part_this))
                this_parts_iou.append(float(I) / float(U))
        this_iou = np.array(this_parts_iou).mean()
        iou_list.append(this_iou)
    
val_miou = np.around(np.array(iou_list).mean()*100, 1)
print(val_miou)


