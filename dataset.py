import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from DeepFakeMask import dfl_full, facehull, components, extended
from PIL import Image
from skimage import io
from skimage import transform as sktransform
from imgaug import augmenters as iaa
import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import json


class FFDataset(Dataset):
    def __init__(self, root_dir, ff_use, compression, use_BI=False, subset_size=3000, candidates=100, type='train'):

        assert type in ['train', 'val', 'test']

        self.root_dir = root_dir

        # 训练集或验证集
        # 按FF++，训练集，验证集，测试集切分为720,140,140
        if type == 'train':
            json_file = os.path.join(self.root_dir, 'splits', 'train.json')
        elif type == 'val':
            json_file = os.path.join(self.root_dir, 'splits', 'val.json')
        else:
            json_file = os.path.join(self.root_dir, 'splits', 'test.json')

        self.data_idx = []
        with open(json_file) as f:
            load_dict = json.load(f)
            for pair in load_dict:
                self.data_idx.append(pair[0])
                self.data_idx.append(pair[1])

        self.ff_datasets = ff_use
        self.compression = compression

        self.type = type if type != 'val' else 'test'

        self.use_blended_images = False if self.type == 'test' else use_BI
        self.subset_size = subset_size
        self.candidates = candidates

        assert self.compression in ['raw', 'c23', 'c40'], \
            'compression needs to be in [\'raw\',\'c23\',\'c40\']'

        dataset_available = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

        if self.ff_datasets == []:
            assert self.use_blended_images, 'you must choose at least one type of fake images'
        else:
            # dataset传入要使用的训练数据集
            # original肯定被使用，这里选择的是假脸数据集
            # 论文中有5种组合
            # 1. FF++四种伪造数据集之一
            # 2. FF++四种伪造数据集之一和BI
            # 3. 单独BI
            # 4. 完整FF++
            # 5. 完整FF++和BI
            assert set(dataset_available) >= set(self.ff_datasets), \
                'dataset needs include one or more of' \
                ' [\'Deepfakes\', \'Face2Face\', \'FaceSwap\', \'NeuralTextures\']'

        # 原始视频帧
        self.real_images = []
        # 伪造视频帧
        self.fake_images = []

        # 读取原始视频帧索引
        real_image_base_path = os.path.join(self.root_dir,
                                            'original_sequences',
                                            'youtube',
                                            self.compression,
                                            'images')
        real_image_dirs = os.listdir(real_image_base_path)

        for dir in real_image_dirs:
            if not dir[:3] in self.data_idx:
                continue
            dir_path = os.path.join(real_image_base_path, dir)
            file_list = os.listdir(dir_path)
            for file in file_list:
                self.real_images.append(str(os.path.join(dir_path, file)).replace('\\', '/'))

        # 读取伪造视频帧索引
        fake_images_base_paths = []
        for dataset_name in self.ff_datasets:
            fake_images_base_paths.append(os.path.join(self.root_dir,
                                                       'manipulated_sequences', dataset_name, self.compression,
                                                       'images'))

        for base_path in fake_images_base_paths:
            fake_image_dirs = os.listdir(base_path)
            for dir in fake_image_dirs:
                if not dir[:3] in self.data_idx:
                    continue
                dir_path = os.path.join(base_path, dir)
                file_list = os.listdir(dir_path)
                for file in file_list:
                    self.fake_images.append(str(os.path.join(dir_path, file)).replace('\\', '/'))

        self.distortion = iaa.Sequential([iaa.PiecewiseAffine(scale=(0.01, 0.15))])
        self.landmarks_dict = {}
        if self.type != 'test':
            self.load_landmarks()

    def load_landmarks(self):
        """
        读取预先提取的视频人脸landmark
        :return:
        """
        DATASET_PATHS = {
            'original': 'original_sequences/youtube',
            # 'Deepfakes': 'manipulated_sequences/Deepfakes',
            # 'Face2Face': 'manipulated_sequences/Face2Face',
            # 'FaceSwap': 'manipulated_sequences/FaceSwap',
            # 'NeuralTextures': 'manipulated_sequences/NeuralTextures'
        }
        print('==> Loading landmarks...')
        for dataset_path in DATASET_PATHS.values():
            landmarks_dir_path = os.path.join(self.root_dir, dataset_path, self.compression, 'landmarks')
            landmarks_path = []
            landmarks_dirs = os.listdir(landmarks_dir_path)

            for dir in landmarks_dirs:
                landmarks_files = os.listdir(os.path.join(landmarks_dir_path, dir))
                for file in landmarks_files:
                    file_path = os.path.join(landmarks_dir_path, dir, file)
                    landmarks_path.append(file_path)

            # 坐标保存形式为：key-文件名;value-人脸坐标
            for landmarks in tqdm(landmarks_path):
                lms = np.load(landmarks)
                image_path = landmarks.replace('landmarks', 'images').replace('npy', 'png')
                self.landmarks_dict[image_path.replace('\\', '/')] = lms

    def get_face_landmarks(self, img_path):
        return self.landmarks_dict[img_path]

    def cal_Eulidean_dist(self, landmark1, landmark2):
        return np.linalg.norm(landmark1.reshape(-1) - landmark2.reshape(-1))

    def nearest_search(self, IB_path, subset_files, candidate_size=100):
        """
        在随机子集中寻找与IB最接近的人脸
        :param IB_path: IB图片路径
        :param subset_files: 随机子集，集合中的元素是文件路径名
        :param candidate_size: 距离最小的n个人脸，从中随机选一个
        :return: 目标人脸IF的文件路径，如果IB没找到人脸（目前使用dlib找人脸，画质和算法原因导致可能出现人脸找不到的情况），就返回None
        """
        IB_landmarks = self.get_face_landmarks(IB_path)

        candidates = []

        # min_dist=9999
        # min_dist_path=''

        for file in subset_files:
            if os.path.dirname(file) == os.path.dirname(IB_path):
                # 来源是同一个视频，跳过
                continue

            # IF_img = cv2.imread(file)
            #
            # IF_face = self.get_dlib_face(IF_img, IF=True)
            # if IF_face is None:
            #     continue

            try:
                IF_landmarks = self.get_face_landmarks(file)
            except:
                continue

            # 计算IB与IF人脸关键点的距离
            dist = self.cal_Eulidean_dist(IB_landmarks, IF_landmarks)
            # print(file)
            candidates.append((file, dist))
            # if dist<min_dist:
            #     min_dist=dist
            #     min_dist_path=file

        candidates.sort(key=lambda x: x[1])
        # 排除掉同视频或大小不同的人脸，可能候选人脸不足candidate_size
        index = random.randint(0, min(candidate_size, len(candidates)) - 1)

        return candidates[index][0]

        # return min_dist_path

    def random_get_hull(self, landmark, img1):
        hull_type = random.choice([0, 1, 2, 3])
        if hull_type == 0:
            mask = dfl_full(landmarks=landmark.astype('int32'), face=img1, channels=3).mask
            return mask / 255
        elif hull_type == 1:
            mask = extended(landmarks=landmark.astype('int32'), face=img1, channels=3).mask
            return mask / 255
        elif hull_type == 2:
            mask = components(landmarks=landmark.astype('int32'), face=img1, channels=3).mask
            return mask / 255
        elif hull_type == 3:
            mask = facehull(landmarks=landmark.astype('int32'), face=img1, channels=3).mask
            return mask / 255

    def random_erode_dilate(self, mask, ksize=None):
        if random.random() > 0.5:
            if ksize is None:
                ksize = random.randint(1, 21)
            if ksize % 2 == 0:
                ksize += 1
            mask = np.array(mask).astype(np.uint8) * 255
            kernel = np.ones((ksize, ksize), np.uint8)
            mask = cv2.erode(mask, kernel, 1) / 255
        else:
            if ksize is None:
                ksize = random.randint(1, 5)
            if ksize % 2 == 0:
                ksize += 1
            mask = np.array(mask).astype(np.uint8) * 255
            kernel = np.ones((ksize, ksize), np.uint8)
            mask = cv2.dilate(mask, kernel, 1) / 255
        return mask

    # borrow from https://github.com/MarekKowalski/FaceSwap
    def blendImages(self, src, dst, mask, featherAmount=0.2):
        maskIndices = np.where(mask != 0)

        src_mask = np.ones_like(mask)
        dst_mask = np.zeros_like(mask)

        maskPts = np.hstack((maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis])).astype(np.int32)
        faceSize = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)
        featherAmount = featherAmount * np.max(faceSize)

        hull = cv2.convexHull(maskPts)
        dists = np.zeros(maskPts.shape[0])
        for i in range(maskPts.shape[0]):
            dists[i] = cv2.pointPolygonTest(hull, (int(maskPts[i, 0]), int(maskPts[i, 1])), True)

        weights = np.clip(dists / featherAmount, 0, 1)

        composedImg = np.copy(dst)
        composedImg[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src[maskIndices[0], maskIndices[1]] + (
                1 - weights[:, np.newaxis]) * dst[maskIndices[0], maskIndices[1]]

        composedMask = np.copy(dst_mask)
        composedMask[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src_mask[
            maskIndices[0], maskIndices[1]] + (
                                                               1 - weights[:, np.newaxis]) * dst_mask[
                                                           maskIndices[0], maskIndices[1]]

        return composedImg, composedMask

    # borrow from https://github.com/MarekKowalski/FaceSwap
    def colorTransfer(self, src, dst, mask):
        transferredDst = np.copy(dst)

        maskIndices = np.where(mask != 0)

        maskedSrc = src[maskIndices[0], maskIndices[1]].astype(np.int32)
        maskedDst = dst[maskIndices[0], maskIndices[1]].astype(np.int32)

        meanSrc = np.mean(maskedSrc, axis=0)
        meanDst = np.mean(maskedDst, axis=0)

        maskedDst = maskedDst - meanDst
        maskedDst = maskedDst + meanSrc
        maskedDst = np.clip(maskedDst, 0, 255)

        transferredDst[maskIndices[0], maskIndices[1]] = maskedDst

        return transferredDst

    def random_Guassian_blur(self, img, kernel_size=None):
        """
        随机kernel_size的高斯模糊
        :param img: 待模糊的图片，ndarray格式
        :param kernel_size:核大小的随机范围，必须是正奇数
        :return:高斯模糊后的图片，ndarray格式
        """
        if kernel_size is None:
            kernel_size = [3, 5, 7, 9]

        # kernel的宽高可以不一样
        kernel_size_w = kernel_size[random.randint(0, len(kernel_size) - 1)]
        kernel_size_h = kernel_size[random.randint(0, len(kernel_size) - 1)]

        blured = cv2.GaussianBlur(img, (9, 9), 5)

        return blured.astype(np.float32)

    def __getitem__(self, index):
        # 用于train的图片全部都是预先将人脸区域截取下来的256*256的图片
        if index < len(self.real_images):
            # 本次加载真人脸
            real_image_path = self.real_images[index]
            face_img = io.imread(real_image_path)
            # 真人脸的mask是全黑的
            mask = np.zeros((face_img.shape[0], face_img.shape[1], 1), dtype=np.uint8)
            cls_target = 0
        else:
            # 本次加载假人脸
            cls_target = 1
            if self.ff_datasets == [] or (self.use_blended_images and random.randint(0, 1) == 0):
                # 使用blend image
                # 如果没有其他的假人脸数据集，则必须使用BI
                IB_index = random.randint(0, len(self.real_images) - 1)
                IB_path = self.real_images[IB_index]

                # 随机大小的子集
                try:
                    subset_index = np.random.permutation(range(0, len(self.real_images)))[:self.subset_size]
                    IF_path = self.nearest_search(IB_path, np.array(self.real_images)[subset_index],
                                                  candidate_size=self.candidates)
                except:
                    return self.__getitem__(index)

                background_face = io.imread(IB_path)
                foreground_face = io.imread(IF_path)

                background_landmark = self.get_face_landmarks(IB_path)

                aug_size = random.randint(128, 317)
                background_landmark = background_landmark * (aug_size / 317)
                foreground_face = sktransform.resize(foreground_face, (aug_size, aug_size), preserve_range=True).astype(
                    np.uint8)
                background_face = sktransform.resize(background_face, (aug_size, aug_size), preserve_range=True).astype(
                    np.uint8)

                mask = self.random_get_hull(background_landmark, background_face)

                #  random deform mask
                mask = self.distortion.augment_image(mask)
                mask = self.random_erode_dilate(mask)

                # filte empty mask after deformation
                if np.sum(mask) == 0:
                    return self.__getitem__(index)
                    # raise NotImplementedError

                # apply color transfer
                foreground_face = self.colorTransfer(background_face, foreground_face, mask * 255)

                # blend two face
                blended_face, mask = self.blendImages(foreground_face, background_face, mask * 255)
                blended_face = blended_face.astype(np.uint8)

                # resize back to default resolution
                blended_face = sktransform.resize(blended_face, (317, 317), preserve_range=True).astype(np.uint8)
                mask = sktransform.resize(mask, (317, 317), preserve_range=True)
                mask = mask[:, :, 0:1]

                mask = (1 - mask) * mask * 4
                face_img = blended_face

            else:
                # 使用数据集提供的假人脸
                index = index - len(self.real_images)
                fake_image_path = self.fake_images[index]
                face_img = io.imread(fake_image_path)
                # 根据文件夹结构找到该视频帧对应的mask文件
                mask = io.imread(str(fake_image_path).replace(self.compression, 'masks'))

                blured_mask = self.random_Guassian_blur(mask) / 255

                # 生成face x-ray
                mask = 4 * blured_mask * (1 - blured_mask)
                mask = np.expand_dims(mask, axis=2)

        # randomly downsample after BI pipeline
        if self.type != 'test' and random.randint(0, 1):
            aug_size = random.randint(64, 317)
            face_img = Image.fromarray(face_img)
            if random.randint(0, 1):
                face_img = face_img.resize((aug_size, aug_size), Image.BILINEAR)
            else:
                face_img = face_img.resize((aug_size, aug_size), Image.NEAREST)
            face_img = face_img.resize((317, 317), Image.BILINEAR)
            face_img = np.array(face_img)

        # random jpeg compression after BI pipeline
        if self.type != 'test' and random.randint(0, 1):
            quality = random.randint(60, 100)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            face_img_encode = cv2.imencode('.jpg', face_img, encode_param)[1]
            face_img = cv2.imdecode(face_img_encode, cv2.IMREAD_COLOR)

        image = face_img[60:316, 30:286, :]
        mask = mask[60:316, 30:286, :]

        # random flip
        if self.type != 'test' and random.randint(0, 1):
            image = np.flip(image, 1)
            mask = np.flip(mask, 1)

        mask = (mask * 255).astype(np.uint8)

        transform = transforms.ToTensor()

        # 返回(视频帧,mask,GT)的元组
        return transform(image.copy()), transform(mask.copy()), torch.Tensor([cls_target])

    def __len__(self):
        len_real = len(self.real_images)
        len_fake = len(self.fake_images)
        # 如果有假人脸图片，则正常返回两个list的长度之和，加载的时候按下标值判断取真还是取假
        # 如果没有假人脸，就返回两倍的真人脸list长度，大于一倍长度的时候使用BI生成假人脸
        return len_real + len_fake if len_fake != 0 else 2 * len_real


def get_train_loader(cfg):
    train_dataset = FFDataset(root_dir=cfg.DATA_DIR,
                              ff_use=cfg.DATASET.TRAIN_SET.FF_USE,
                              compression=cfg.DATASET.TRAIN_SET.COMPRESSION,
                              use_BI=cfg.DATASET.TRAIN_SET.BLENDED_IMAGES.USE,
                              subset_size=cfg.DATASET.TRAIN_SET.BLENDED_IMAGES.SUBSET_SIZE,
                              candidates=cfg.DATASET.TRAIN_SET.BLENDED_IMAGES.CANDIDATES,
                              type='train')
    return DataLoader(dataset=train_dataset,
                      batch_size=cfg.BATCH_SIZE,
                      shuffle=True,
                      num_workers=cfg.WORKERS,
                      drop_last=True)


def get_val_loader(cfg):
    train_dataset = FFDataset(root_dir=cfg.DATA_DIR,
                              ff_use=cfg.DATASET.VAL_SET.FF_USE,
                              compression=cfg.DATASET.VAL_SET.COMPRESSION,
                              type='train')
    return DataLoader(dataset=train_dataset,
                      batch_size=cfg.BATCH_SIZE,
                      shuffle=False,
                      num_workers=cfg.WORKERS,
                      drop_last=False)


def get_test_loader(root_dir, test_dataset, compression=None, batch_size=32, workers=16):
    ff_dataset = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    if test_dataset[0] in ff_dataset:
        test_set = FFDataset(root_dir=root_dir,
                             ff_use=test_dataset,
                             compression=compression,
                             type='test')
    elif len(test_dataset) == 1 and test_dataset[0] == 'celeb-df':
        test_set = ImageFolder(root=root_dir, transform=transforms.ToTensor())

    return DataLoader(dataset=test_set,
                      batch_size=batch_size,
                      shuffle=False,
                      num_workers=workers,
                      drop_last=False)
