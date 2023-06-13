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
ckpt_path = os.path.join(ckpt_root, 'reggeonet_cls_sonn_bg.pth')


train_bs = 64
train_set = SONNBGClsParaLoader(data_root, 'train')
train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True, num_workers=8, worker_init_fn=seed_worker, drop_last=True)
test_bs = 64
test_set = SONNBGClsParaLoader(data_root, 'test')
test_loader = DataLoader(test_set, batch_size=test_bs, shuffle=False, num_workers=8, worker_init_fn=seed_worker, drop_last=False)


net = RegGeoNetClsSparse(num_classes=15).cuda()
max_lr = 5e-2
min_lr = 5e-4
num_epc = 500
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=max_lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epc, eta_min=min_lr)


best_test_acc = 0
for epc in range(1, num_epc+1):
    
    net.train()
    epoch_loss = 0
    num_samples = 0
    num_correct = 0
    for (gi_img, cid) in tqdm(train_loader):
        optimizer.zero_grad()
        gi_img = gi_img.cuda()
        cid = cid.long().cuda()
        bs = gi_img.size(0)
        logits = net(gi_img)
        loss = compute_smooth_cross_entropy(logits, cid, 0.2)
        loss.backward()
        optimizer.step()
        preds = logits.argmax(dim=-1).detach()
        num_samples += bs
        num_correct += (preds==cid).sum().item()
        epoch_loss += (loss.item() * bs)
    scheduler.step()
    epoch_loss = np.around(epoch_loss/num_samples, 5)
    train_acc = np.around((num_correct/num_samples)*100, 1)
    print('epoch: {}: train acc: {}%, loss: {}'.format(epc, train_acc, epoch_loss))
    
    cond_1 = (train_acc<=80.0 and np.mod(epc, 5)==0)
    cond_2 = (train_acc>=80.0 and train_acc<=90.0 and np.mod(epc, 2)==0)
    cond_3 = (train_acc>=90.0 and np.mod(epc, 1)==0)
    if epc<=3 or cond_1 or cond_2 or cond_3:
        net.eval()
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
        test_acc = np.around((num_correct/num_samples)*100, 1)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save({'model': net.state_dict(), 'optim': optimizer.state_dict()}, ckpt_path)
        print('epoch: {}: test acc: {}%,  best test acc: {}%'.format(epc, test_acc, best_test_acc))


