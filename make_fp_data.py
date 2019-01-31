import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from sklearn import metrics
from torch import optim

from data_loader import DataProvider
from model import InceptionV3


def val(model: nn.Module, criterion, val_loader: data_utils.DataLoader, epoch):
    model.eval()

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

            loss = criterion(outputs.view(-1, 2), label)

            # if i % 10 == 0:
    print()
    # niter = epoch * len(val_loader) + i
    # writer.add_scalar('Train/val_oss', losses.avg, epoch)
    pred_arr = np.concatenate(pred_arr)
    label_arr = np.concatenate(label_arr)
    cfm = metrics.confusion_matrix(label_arr, pred_arr)
    print(cfm)
    return top1.avg


def main():
    train_loader = data_utils.DataLoader(dataset=DataProvider(),

                                         batch_size=120, num_workers=18, worker_init_fn=worker_init_fn)
    val_loader = data_utils.DataLoader(dataset=DataProvider(val=True),
                                       batch_size=120, num_workers=18, worker_init_fn=worker_init_fn)
    best_acc = 0
    model = InceptionV3().cuda()
    model = nn.DataParallel(model)
    # optimizer = optim.Adam(model.module.parameters(), lr=1e-4)
    optimizer = optim.RMSprop(model.module.parameters(), lr=0.05 / 2, momentum=0.9, weight_decay=0.5)

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

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(2e6 / len(train_loader)), gamma=0.5, )
    # last_epoch=start_epoch)
    for epoch in range(start_epoch, 500):
        lr_scheduler.step()
        np.random.seed()
        # train(model, optimizer, criterion, train_loader, epoch)
        val(model, criterion, val_loader, epoch)


if __name__ == '__main__':
    main()
