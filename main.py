import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as data_utils
from data_loader import DataProvider
from model import VGG_FCN
import shutil


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
    for i, (img, label) in enumerate(train_loader):
        img, label = img.cuda(), label.cuda()
        out = model(img)
        # print(out.shape, label.shape)
        acc = accuracy(out, label)
        loss = criterion(out.view(-1, 2), label)

        losses.update(loss.item(), img.size(0))
        top1.update(acc, img.shape[0])

        optim.zero_grad()
        loss.backward()
        optim.step()
        if i % 10 == 0:
            print(f'\rEpoch: {epoch} loss: {losses.avg:.4f}, {top1.avg:.4f}', end='')
    print()


def val(model: nn.Module, criterion, val_loader: data_utils.DataLoader, epoch):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    with torch.no_grad():
        for i, (img, label) in enumerate(val_loader):
            img, label = img.cuda(), label.cuda()
            out = model(img)

            acc = accuracy(out, label)
            loss = criterion(out.view(-1, 2), label)

            losses.update(loss.item(), img.size(0))
            top1.update(acc, img.shape[0])
            if i % 10 == 0:
                print(f'\rEpoch: {epoch}  {losses.avg:.4f},  {top1.avg:.4f}', end='')
    print()
    return top1.avg


def main():
    train_loader = data_utils.DataLoader(dataset=DataProvider(),

                                         batch_size=240, num_workers=40)
    val_loader = data_utils.DataLoader(dataset=DataProvider(val=True),
                                       batch_size=240, num_workers=40)
    best_acc = 0
    model = VGG_FCN().cuda()
    model = nn.DataParallel(model)
    optimizer = optim.Adam(model.module.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(100):
        train(model, optimizer, criterion, train_loader, epoch)
        acc = val(model, criterion, val_loader, epoch)
        is_best = acc > best_acc

        if epoch % 2 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                # 'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best)


if __name__ == '__main__':
    main()
