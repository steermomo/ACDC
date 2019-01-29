import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as data_utils
from data_loader import DataProvider
from model import VGG_FCN
import shutil
from sklearn import metrics
import numpy as np
import torchvision.utils as vutils
import os
from tensorboardX import SummaryWriter

writer = SummaryWriter('log_model2')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.view(1, -1)
        correct = pred.eq(target.view(1, -1))

        correct_k = correct.view(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size).item()


def train(model: nn.Module, optim, criterion, train_loader: data_utils.DataLoader, epoch):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    pred_arr = []
    label_arr = []
    for i, (img, label) in enumerate(train_loader):
        print(f'\rEpoch: {epoch} {i} / {len(train_loader)}', end='')
        label_arr.append(label.numpy())
        img, label = img.cuda(), label.cuda()
        # out = model(img)

        # inception
        outputs = model(img)
        # print(outputs.shape)
            loss = criterion(outputs.view(-1, 2), label)
        # loss2 = criterion(aux_outputs, label)
        # loss = loss1 + 0.4 * loss2

        _, pred = outputs.topk(1, 1, True, True)
        pred = pred.view(-1).detach().cpu().numpy()
        pred_arr.append(pred)

        # print(out.shape, label.shape)
        acc = accuracy(outputs, label)
        # loss = criterion(outputs.view(-1, 2), label)

        losses.update(loss.item(), img.size(0))
        top1.update(acc, img.shape[0])

        optim.zero_grad()
        loss.backward()
        optim.step()
        # if i % 10 == 0:
        print(f' loss: {losses.avg:.4f}, acc: {top1.avg:.4f}', end='')
        if i == 0:
            niter = epoch * len(train_loader) + i
            x = vutils.make_grid(img)
            writer.add_image('Train_Image', x, niter)
    # niter = epoch * len(train_loader) + i
    writer.add_scalar('Train/train_loss', losses.avg, epoch)
    print()
    pred_arr = np.concatenate(pred_arr)
    label_arr = np.concatenate(label_arr)
    cfm = metrics.confusion_matrix(label_arr, pred_arr)
    print(cfm)


def val(model: nn.Module, criterion, val_loader: data_utils.DataLoader, epoch):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    pred_arr = []
    label_arr = []
    with torch.no_grad():
        for i, (img, label) in enumerate(val_loader):
            print(f'\rEpoch: {epoch} {i} / {len(val_loader)}', end='')
            label_arr.append(label.numpy())

            img, label = img.cuda(), label.cuda()
            # out = model(img)
            outputs = model(img)
            # loss1 = criterion(outputs, label)
            # loss2 = criterion(aux_outputs, label)
            # loss = loss1 + 0.4 * loss2

            _, pred = outputs.topk(1, 1, True, True)
            pred = pred.view(-1).detach().cpu().numpy()
            pred_arr.append(pred)

            acc = accuracy(outputs, label)
            loss = criterion(outputs.view(-1, 2), label)

            losses.update(loss.item(), img.size(0))
            top1.update(acc, img.shape[0])
            # if i % 10 == 0:
            print(f' val_loss:  {losses.avg:.4f}, acc: {top1.avg:.4f}', end='')

            if i == 0:
                niter = epoch * len(val_loader) + i
                x = vutils.make_grid(img)
                writer.add_image('Val_Image', x, niter)
    print()
    # niter = epoch * len(val_loader) + i
    writer.add_scalar('Train/val_oss', losses.avg, epoch)
    pred_arr = np.concatenate(pred_arr)
    label_arr = np.concatenate(label_arr)
    cfm = metrics.confusion_matrix(label_arr, pred_arr)
    print(cfm)
    return top1.avg


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def main():
    gpu_num = torch.cuda.device_count()
    train_loader = data_utils.DataLoader(dataset=DataProvider(),

                                         batch_size=60 * gpu_num, num_workers=18, worker_init_fn=worker_init_fn)
    val_loader = data_utils.DataLoader(dataset=DataProvider(val=True),
                                       batch_size=60 * gpu_num, num_workers=18, worker_init_fn=worker_init_fn)
    best_acc = 0
    model = VGG_FCN().cuda()
    model = nn.DataParallel(model)
    # optimizer = optim.Adam(model.module.parameters(), lr=1e-4)
    optimizer = optim.RMSprop(model.module.parameters(), lr=0.05 / 2, alpha=0.9, eps=1.0, momentum=0.9, )  #weight_decay=0.5)

    criterion = nn.CrossEntropyLoss()
    start_epoch = 0
    resume = 'model_best.pth.tar'
    if os.path.isfile(resume):
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(2e6 / len(train_loader)), gamma=0.5,)
                                             # last_epoch=start_epoch)
    for epoch in range(start_epoch, 500):
        lr_scheduler.step()
        np.random.seed()
        train(model, optimizer, criterion, train_loader, epoch)
        acc = val(model, criterion, val_loader, epoch)
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if epoch % 1 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                # 'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best)


if __name__ == '__main__':
    main()
