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


snp_objects_names, snp_objects_parts = ShapeNetPart_ObjectsParts()
train_bs = 32
train_set = SNPSegParaLoader(data_root, 'train')
train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True, num_workers=8, worker_init_fn=seed_worker, drop_last=True)
test_bs = 12
test_set = SNPSegParaLoader(data_root, 'test')
test_loader = DataLoader(test_set, batch_size=test_bs, shuffle=False, num_workers=8, worker_init_fn=seed_worker, drop_last=False)


net = RegGeoNetSeg(num_object_classes=16, num_part_classes=50).cuda()
max_lr = 1e-1
min_lr = 1e-3
num_epc = 120
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=max_lr, momentum=0.9, weight_decay=1e-6)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epc, eta_min=min_lr)


m = 512
M = (m ** 2)
best_val_miou = 0
for epc in range(1, num_epc+1):
    net.train()
    epoch_loss = 0
    num_samples = 0
    for (G, L_gt, cid, name_list) in tqdm(train_loader):
        optimizer.zero_grad()
        G = G.cuda()
        L_gt = L_gt.long().cuda()
        cid = cid.long().cuda()
        bs = G.size(0)
        logits = net(G, cid)
        logits_reshaped = logits.view(bs, -1, M).permute(0, 2, 1).contiguous()
        L_gt_reshaped = L_gt.squeeze(1).view(bs, M)
        loss = compute_smooth_cross_entropy(logits_reshaped.view(bs*M, -1), L_gt.view(bs*M), eps=0.02)
        loss.backward()
        optimizer.step()
        num_samples += bs
        epoch_loss += (loss.item() * bs)
    scheduler.step()
    epoch_loss = np.around(epoch_loss/num_samples, 5)
    print('epoch: {}, seg loss: {}'.format(epc, epoch_loss))
    
    cond_1 = (epc<=int(num_epc*0.90) and np.mod(epc, 5)==0)
    cond_2 = (epc>=int(num_epc*0.90) and np.mod(epc, 1)==0)
    if epc<=3 or cond_1 or cond_2:
        net.eval()
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
        val_miou = np.around(np.array(iou_list).mean()*100, 2)
        if val_miou >= best_val_miou:
            best_val_miou = val_miou
            torch.save(net.state_dict(), ckpt_path)
        print('epoch: {}: val miou: {}%,  best val miou: {}%'.format(epc, val_miou, best_val_miou))

        
        