from torch import nn
import torch.nn.functional as F
from dataset import get_test_loader
from HRNet import get_net
from tqdm import tqdm
import argparse
import time
import pprint
from utils import *
from torchmetrics import AUROC


def test(model, test_loader, auc, criterion_cls, criterion_mask=None, lamb=100):
    """
    跑测试集
    :param model: 模型
    :param test_loader: 测试数据loader
    :param auc: torchmetrics工具，用于计算auc
    :param criterion_cls: 二分类的损失函数
    :param criterion_mask: mask的损失函数
    :param lamb: 论文中的λ
    :return: 平均损失、auc、平均精度
    """

    model.eval()

    loss = AverageMeter('loss')
    acc_meter = AverageMeter('acc')
    loop = tqdm(enumerate(test_loader), total=len(test_loader))

    with torch.no_grad():
        for batch_idx, data in loop:
            if len(data) == 3:
                (image, mask, cls_target) = data
                image = image.cuda()
                mask = mask.flatten().cuda()
                cls_target = cls_target.flatten().cuda()
                mask_predict, cls_predict = model(image)
                mask_predict = mask_predict.flatten()

                loss_mask = criterion_mask(mask_predict, mask)
                loss_cls = criterion_cls(cls_predict, cls_target.long())
                loss_cur = loss_mask * lamb + loss_cls

            elif len(data) == 2:
                (image, cls_target) = data
                image = image.cuda()
                cls_target = cls_target.flatten().cuda()
                _, cls_predict = model(image)
                loss_cur = criterion_cls(cls_predict, cls_target.long())

            # 计算平均loss
            n = image.size(0)
            loss.update(loss_cur.item(), n)

            # 计算分类精度
            cls_softmax = F.softmax(cls_predict, dim=1).data
            _, predicted = torch.max(cls_softmax, dim=1)
            cls_target = cls_target.flatten()
            correct = (predicted == cls_target).sum().item()
            acc = 100 * correct / n
            acc_meter.update(acc, n)

            # auc
            auc.update(cls_softmax, cls_target.long())
            auc_cur = auc.compute().item() * 100

            loop.set_description('Testing')
            loop.set_postfix(loss=f'{loss.avg:.4f}', AUC=f'{auc_cur:.3f}',
                             acc=f'{acc_meter.avg:.3f}')

    return loss.avg, auc_cur, acc_meter.avg


def main(args):
    # 打开日志文件
    time_str = time.strftime('%Y-%m-%d_%H_%M')
    logger_name = f'test_logger{time_str}.log'
    print_logger = get_logger(os.path.join(args.output_dir, logger_name))

    print_logger.info(pprint.pformat(args))

    # 加载HRNet模型
    print_logger.info('==> loading HRNet...')
    devices = try_all_gpus()
    net = get_net(cfg_file='HRNet/hrnet_config/experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml',
                  devices=devices)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    # 加载测试数据
    print_logger.info('==> Preparing data...')
    test_loader = get_test_loader(root_dir=args.root_dir,
                                  test_dataset=args.test_data,
                                  compression=args.compression,
                                  batch_size=args.batch_size,
                                  workers=args.workers)

    # 加载评估模型
    checkpoint_file = os.path.join(args.ckpt_dir, 'best_model.pth.tar')
    print_logger.info(f'==> Loading best model from {checkpoint_file}')
    new_state_dict, _, _, best_auc, start_epoch = load_checkpoint(checkpoint_file, devices)
    net.load_state_dict(new_state_dict)
    print_logger.info(f'==> Loading best model success, best AUC:{best_auc:.3f}, epoch:{start_epoch - 1}')

    print_logger.info('==> Start testing')
    print_logger.info(f'==> Testing: {args.test_data}')
    auc = AUROC(num_classes=2)
    criterion_mask = nn.BCELoss()
    criterion_cls = nn.CrossEntropyLoss()
    test_loss, test_auc, test_acc = test(net, test_loader, auc, criterion_cls, criterion_mask)
    print_logger.info(f'==> Testing end. Test AUC:{test_auc:.3f}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hrnet_model', type=str, default='./HRNet/pretrained/hrnetv2_pretrained.pth',
                        help='HRNet pretrained model')
    parser.add_argument('--ckpt_dir', type=str, help='Directory of checkpoint')
    parser.add_argument('-d', '--test_data', nargs='+', help='Test dataset')
    parser.add_argument('-c', '--compression', type=str, default='raw', help='Compression')
    parser.add_argument('-r', '--root_dir', type=str, default='./dataset/FaceForensics++',
                        help='Root directory of dataset')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--workers', type=int, default=16, help='num_workers')
    parser.add_argument('-o', '--output_dir', type=str, default='./log', help='Output directory of log')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
