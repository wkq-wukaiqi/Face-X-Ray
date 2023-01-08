from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from config import cfg, update_config
from dataset import get_train_loader, get_val_loader
from HRNet import get_net
from tqdm import tqdm
from torchmetrics import AUROC
import argparse
import time
import pprint
from utils import *

DATA_SET = {
    'Deepfakes': 'DF',
    'Face2Face': 'F2F',
    'FaceSwap': 'FS',
    'NeuralTextures': 'NT'
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, default='./experiments/default.yaml',
                        help='Experimental parameters in YAML format')
    parser.add_argument('--hrnet_model', type=str, default='./HRNet/pretrained/hrnetv2_pretrained.pth',
                        help='HRNet pretrained model')
    parser.add_argument('-r', '--resume_dir', type=str,
                        help='The directory of checkpoint to resume')

    args = parser.parse_args()
    return args


def train(cfg, epoch, model, train_loader, criterion_mask, criterion_cls, optimizer, auc):
    """
    训练一轮
    :param cfg: 配置文件对象
    :param epoch: 轮次
    :param model: 模型
    :param train_loader: 训练数据loader
    :param criterion_mask: mask的损失函数
    :param criterion_cls: 二分类的损失函数
    :param optimizer: 优化器
    :param auc: torchmetrics工具，用于计算auc
    :return: 平均loss、auc
    """
    model.train()
    loop = tqdm(enumerate(train_loader), total=len(train_loader))

    loss = AverageMeter('loss')
    acc_meter = AverageMeter('acc')

    for batch_idx, (image, mask, cls_target) in loop:
        image = image.cuda()
        mask = mask.flatten().cuda()
        cls_target = cls_target.flatten().cuda()

        mask_predict, cls_predict = model(image)
        mask_predict = mask_predict.flatten()

        loss_mask = criterion_mask(mask_predict, mask)
        loss_cls = criterion_cls(cls_predict, cls_target.long())
        loss_cur = loss_mask * cfg.TRAIN.LAMBDA + loss_cls

        optimizer.zero_grad()
        loss_cur.backward()
        optimizer.step()

        # 计算平均loss
        n = image.size(0)
        loss.update(loss_cur.item(), n)

        cls_softmax = F.softmax(cls_predict, dim=1).data

        # acc
        _, predicted = torch.max(cls_softmax, dim=1)
        cls_target = cls_target.flatten()
        correct = (predicted == cls_target).sum().item()
        acc = 100 * correct / n
        acc_meter.update(acc, n)

        # AUC
        auc.update(cls_softmax, cls_target.long())
        auc_cur = auc.compute().item() * 100

        loop.set_description(f'Train Epoch:[{epoch}/{cfg.TRAIN.TOTAL_EPOCHS}]')
        loop.set_postfix(loss=f'{loss.avg:.4f}', AUC=f'{auc_cur:.3f}', acc=f'{acc_meter.avg:.3f}')

    return loss.avg, auc_cur


def validate(epoch, model, test_loader, criterion_mask, criterion_cls, auc):
    """
    验证
    :param epoch: 轮次
    :param model: 模型
    :param test_loader: 训练数据loader
    :param criterion_mask: mask的损失函数
    :param criterion_cls: 二分类的损失函数
    :param auc: torchmetrics，用于计算auc
    :return:
    """

    model.eval()

    loss = AverageMeter('loss')
    acc_meter = AverageMeter('acc')
    loop = tqdm(enumerate(test_loader), total=len(test_loader))

    with torch.no_grad():
        for batch_idx, (image, mask, cls_target) in loop:
            image = image.cuda()
            mask = mask.flatten().cuda()
            cls_target = cls_target.flatten().cuda()
            mask_predict, cls_predict = model(image)
            mask_predict = mask_predict.flatten()

            loss_mask = criterion_mask(mask_predict, mask)
            loss_cls = criterion_cls(cls_predict, cls_target.long())
            loss_cur = loss_mask * cfg.TRAIN.LAMBDA + loss_cls

            # 计算平均loss
            n = image.size(0)
            loss.update(loss_cur.item(), n)

            cls_softmax = F.softmax(cls_predict, dim=1).data

            _, predicted = torch.max(cls_softmax, dim=1)
            cls_target = cls_target.flatten()
            correct = (predicted == cls_target).sum().item()
            acc = 100 * correct / n
            acc_meter.update(acc, n)

            auc.update(cls_softmax, cls_target.long())
            auc_cur = auc.compute().item() * 100

            loop.set_description(f'Test Epoch:[{epoch}/{cfg.TRAIN.TOTAL_EPOCHS}]')
            loop.set_postfix(loss=f'{loss.avg:.4f}', AUC=f'{auc_cur:.3f}',
                             acc=f'{acc_meter.avg:.3f}')

    return loss.avg, auc_cur, acc_meter.avg


def main(args):
    config_file = args.config_file
    # 读取配置文件
    update_config(cfg, config_file)
    # 打开日志文件
    time_str = time.strftime('%Y-%m-%d_%H_%M')
    logger_name = f'train_logger{time_str}_{os.path.basename(config_file)[:-5]}.log'
    print_logger = get_logger(os.path.join(cfg.LOG_DIR, logger_name))

    output_dir = cfg.OUTPUT_DIR + '_' + time_str

    print_logger.info(pprint.pformat(args))
    print_logger.info(pprint.pformat(cfg))

    # 加载HRNet预训练模型
    print_logger.info('==> loading HRNet...')
    devices = try_all_gpus()
    net = get_net(cfg_file='HRNet/hrnet_config/experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml',
                  devices=devices,
                  state_dict_path=args.hrnet_model)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    # 加载数据
    print_logger.info('==> Preparing data...')
    train_loader = get_train_loader(cfg)
    val_loader = get_val_loader(cfg)

    net.HRNet_layer.requires_grad_(False)
    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, net.parameters()),
                           lr=cfg.TRAIN.LR)
    scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer,
                                         milestones=cfg.TRAIN.LR_STEP,
                                         gamma=cfg.TRAIN.GAMMA)

    criterion_mask = nn.BCELoss()
    criterion_cls = nn.CrossEntropyLoss()

    start_epoch = 1
    best_auc = 0.0
    optim_state_dict = None
    sche_state_dict = None

    # 加载检查点
    if args.resume_dir:
        checkpoint_file = os.path.join(args.resume_dir, 'checkpoint.pth.tar')
        print_logger.info(f'==> Resuming checkpoint from {checkpoint_file}')
        new_state_dict, optim_state_dict, sche_state_dict, best_auc, start_epoch = load_checkpoint(checkpoint_file,
                                                                                                   devices)
        net.load_state_dict(new_state_dict)
        print_logger.info(f'==> Resuming checkpoint success, best AUC:{best_auc:.3f}, latest epoch:{start_epoch - 1}')

    freeze = True
    # 若warm轮次达到，解冻参数
    if start_epoch > cfg.TRAIN.WARM_UP_EPOCHS:
        print_logger.info('==> Warm up end, unfreezing all layers')
        freeze = False
        net.HRNet_layer.requires_grad_(True)
        optimizer.add_param_group({'params': net.HRNet_layer.parameters()})

    # 检查点中优化器、lr_sche的参数
    if optim_state_dict:
        optimizer.load_state_dict(optim_state_dict)
    if sche_state_dict:
        scheduler.load_state_dict(sche_state_dict)

    epoch = start_epoch
    auc = AUROC(num_classes=2)
    # 训练+验证
    while epoch <= cfg.TRAIN.TOTAL_EPOCHS:
        cur_lr = optimizer.param_groups[0]['lr']
        print_logger.info(f'Epoch:[{epoch}/{cfg.TRAIN.TOTAL_EPOCHS}] lr:{cur_lr}')

        # 训练一轮
        train_loss, train_auc = train(cfg, epoch, net, train_loader, criterion_mask, criterion_cls, optimizer, auc)
        print_logger.info(f'==> Train loss:{train_loss:.3f} AUC:{train_auc :.3f}')
        scheduler.step()
        auc.reset()

        # 验证一轮
        test_loss, test_auc, test_acc = validate(epoch, net, val_loader, criterion_mask, criterion_cls, auc)
        auc.reset()
        print_logger.info(f'==> Validate loss:{test_loss:.3f} AUC:{test_auc :.3f}')

        # 更新最佳验证auc
        is_best = False
        if test_auc > best_auc:
            best_auc = test_auc
            is_best = True

        epoch += 1

        if epoch > cfg.TRAIN.WARM_UP_EPOCHS and freeze:
            # warmup结束，解冻
            print_logger.info('==> Warm up end, unfreezing all layers')
            net.HRNet_layer.requires_grad_(True)
            optimizer.add_param_group({'params': net.HRNet_layer.parameters()})
            freeze = False

        save_checkpoint({
            'epoch': epoch,
            'best_auc': best_auc,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, is_best, output_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)
