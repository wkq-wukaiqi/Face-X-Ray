import torch
from collections import OrderedDict
import os
import shutil
import logging


def get_logger(file_path):
    """
    获取日志对象
    :param file_path: 日志文件地址
    :return: 日志对象
    """
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    logger = logging.getLogger()

    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %H:%M:%S')

    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.setLevel(logging.INFO)

    return logger


def try_all_gpus():
    """
    获取所有能用的gpu
    :return: devices列表
    """
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu'), ]


def save_checkpoint(state, is_best, save_dir):
    """
    保存检查点
    :param state: 检查点信息
    :param is_best: 是否是验证集最佳
    :param save_dir: 保存的文件夹
    :return:
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 保存检查点
    filename = os.path.join(save_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    # 保存最佳模型
    if is_best:
        best_file = os.path.join(save_dir, 'best_model.pth.tar')
        shutil.copy(filename, best_file)


def load_checkpoint(path, devices):
    """
    加载检查点
    :param path: 检查点文件路径
    :param devices: devices列表
    :return: 模型参数字典、优化器参数字典、lr_sche字典、最佳auc、开始训练的轮次
    """
    checkpoint = torch.load(path)
    best_auc = checkpoint['best_auc']
    start_epoch = checkpoint['epoch'] + 1

    new_state_dict = OrderedDict()
    old_state_dict = checkpoint['state_dict']
    # 如果是多卡参数名前面会多一个 module. 这里对这个问题进行处理
    if len(devices) > 1:
        for k, v in old_state_dict.items():
            new_state_dict['module.' + k.replace('module.', '')] = v
    else:
        for k, v in old_state_dict.items():
            new_state_dict[k.replace('module.', '')] = v

    optim_state_dict = checkpoint['optimizer']
    sche_state_dict = checkpoint['scheduler']

    return new_state_dict, optim_state_dict, sche_state_dict, best_auc, start_epoch


class AverageMeter(object):
    """
    计算均值的工具类
    """

    def __init__(self, name, fmt=':.3f'):
        self.name = name
        self.fmt = fmt

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return 'name:{} latest_val:{' + self.fmt + '} avg:{' + self.fmt + '}'.format(self.name, self.val, self.avg)
