import os
from os.path import join
import cv2
from tqdm import tqdm
import dlib
import numpy as np
import json
import argparse

# 子数据集下载默认路径
DATASET_PATHS = {
    'original': 'original_sequences/youtube',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures'
}

# dlib模型路径
DLIB_MODEL_PATH = './dlib_model/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DLIB_MODEL_PATH)


def get_dlib_face(img):
    """
    调用dlib获取人脸位置
    :param img: 待截取人脸的图片
    :return: 人脸位置对象face，包括(face.left(), face.top()), (face.right(), face.bottom())
    """
    faces = detector(img, 0)
    if len(faces) == 0:
        return None
    else:
        return faces[0]


def get_face_landmarks(img, face):
    """
    获取图片中的人脸坐标，FF++数据集大多数人脸只有一个，且在正面
    :param img: 图片
    :param face: dlib获取的人脸框位置
    :return: 人脸68特征点坐标，形状为(68,2)，格式为numpy数组
    """
    shape = predictor(img, face)
    # 将dlib检测到的人脸特征点转为numpy格式
    res = np.zeros((68, 2), dtype=int)
    for i in range(0, 68):
        # 68个特征点
        res[i] = np.array([shape.part(i).x, shape.part(i).y])

    return res


def conservative_crop(img, face, scale=2.0, new_size=(317, 317)):
    """
    FF++论文中裁剪人脸是先找到人脸框，然后人脸框按比例扩大后裁下更大区域的人脸
    :param img: 待裁剪人脸图片
    :param face: dlib获取的人脸区域
    :param scale: 扩大比例
    :return: 裁剪下来的人脸区域，大小默认为(256,256)，Implementation detail中说预测的mask上采样为256*256，所以截取的人脸应该也是这个大小
    """

    height, width = img.shape[:2]

    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    cropped = cv2.resize(img[y1:y1 + size_bb, x1:x1 + size_bb, :], new_size)

    return cropped


def get_mask(img):
    _, binary_b = cv2.threshold(img[:, :, 0], 0, 255, cv2.THRESH_BINARY)
    _, binary_g = cv2.threshold(img[:, :, 1], 0, 255, cv2.THRESH_BINARY)
    _, binary_r = cv2.threshold(img[:, :, 2], 0, 255, cv2.THRESH_BINARY)
    mask = np.clip(binary_b + binary_g + binary_r, 0, 255)

    res = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in range(len(contours)):
        # 是否为凸包
        ret = cv2.isContourConvex(contours[c])
        # 凸包检测
        points = cv2.convexHull(contours[c])

        # 返回的points形状为(凸包边界点个数，1,2)
        # 使用fillPoly函数应该把前两个维度对调
        points = np.transpose(points, (1, 0, 2))
        # print(points.shape)
        # 描点然后用指定颜色填充点围成的图形内部，生成原始的mask
        cv2.fillPoly(res, points, color=(255, 255, 255))

    return np.expand_dims(res, axis=2)


def extract_frames(data_path, output_path, video_type, compression='raw', is_train=True):
    os.makedirs(output_path, exist_ok=True)
    reader = cv2.VideoCapture(data_path)

    reader_mask = None

    if video_type != 'original':
        data_path_mask = data_path.replace(compression, 'masks')
        output_path_mask = output_path.replace(compression, 'masks')
        os.makedirs(output_path_mask, exist_ok=True)
        reader_mask = cv2.VideoCapture(data_path_mask)

    face_miss_count = 0
    frame_num = 0
    i = 0
    while reader.isOpened() and (reader_mask is None or reader_mask.isOpened()):
        # 人脸mask一起读，并且帧数对齐
        success, image = reader.read()
        if reader_mask is not None:
            success_mask, mask = reader_mask.read()
        if not success:
            break
        if reader_mask is not None and not success_mask:
            break
        i += 1
        # 每四帧提取一次
        if is_train and i % 4 != 0:
            continue
        # 用dlib获取人脸
        face = get_dlib_face(image)
        if face is not None:
            # 截取人脸部分
            image = conservative_crop(image, face)
            if video_type == 'original':
                # 原始视频帧
                if is_train:
                    # 用于训练的视频，在截取人脸后的图片上提前提取人脸坐标并保存
                    face = get_dlib_face(image)
                    if face is None:
                        continue
                    lms = get_face_landmarks(image, face)
                    # 预先保存人脸坐标，用于生成blend image
                    dest_path = join(output_path.replace('images', 'landmarks'), '{:04d}.npy'.format(frame_num))
                    if not os.path.exists(os.path.dirname(dest_path)):
                        os.makedirs(os.path.dirname(dest_path))
                    np.save(dest_path, lms)
                # 保存截取的人脸，测试视频只需要保存人脸
                cv2.imwrite(join(output_path, '{:04d}.png'.format(frame_num)),
                            image)
            elif video_type == 'NeuralTextures':
                # NeuralTextures伪造人脸
                # 在截取人脸后的图片上再次提取人脸
                face = get_dlib_face(image)
                if face is not None:
                    # 保存人脸部分的图片
                    cv2.imwrite(join(output_path, '{:04d}.png'.format(frame_num)),
                                image)
                    # 生成NeuralTextures数据集使用的UV mask
                    landmarks = get_face_landmarks(image, face)
                    mask = np.zeros(image.shape[0:2] + (1,), dtype=np.float32)
                    uv_region = (landmarks[1:16],
                                 landmarks[29:36],
                                 landmarks[50:68])
                    parts = [uv_region]
                    for item in parts:
                        merged = np.concatenate(item)
                        cv2.fillConvexPoly(mask, cv2.convexHull(merged), 255.)  # pylint: disable=no-member
                    # 保存伪造mask
                    cv2.imwrite(join(output_path_mask, '{:04d}.png'.format(frame_num)),
                                mask)
            else:
                # Deepfakes、Face2Face、FaceSwap数据集
                # 保存视频截取的人脸
                cv2.imwrite(join(output_path, '{:04d}.png'.format(frame_num)),
                            image)
                # 参照人脸坐标截取mask
                mask = conservative_crop(mask, face)
                mask = get_mask(mask)
                # 保存mask
                cv2.imwrite(join(output_path_mask, '{:04d}.png'.format(frame_num)),
                            mask)
        else:
            face_miss_count += 1
        frame_num += 1
    reader.release()
    return frame_num, face_miss_count


def get_split_index(split_dir):
    train_idx = []
    val_idx = []
    test_idx = []

    with open(os.path.join(split_dir, 'train.json')) as f:
        load_dict = json.load(f)
        for pair in load_dict:
            train_idx.append(pair[0])
            train_idx.append(pair[1])

    with open(os.path.join(split_dir, 'val.json')) as f:
        load_dict = json.load(f)
        for pair in load_dict:
            val_idx.append(pair[0])
            val_idx.append(pair[1])

    with open(os.path.join(split_dir, 'test.json')) as f:
        load_dict = json.load(f)
        for pair in load_dict:
            test_idx.append(pair[0])
            test_idx.append(pair[1])

    return train_idx, val_idx, test_idx


def parse_args():
    parser = argparse.ArgumentParser()
    # 数据集根目录
    parser.add_argument('-d', '--data_path', type=str, default='./',
                        help='FF++ dataset root path')
    # 提取人脸输出文件夹
    parser.add_argument('-o', '--output_path', type=str, default='./extract',
                        help='output path')
    # 训练、验证、测试切分文件夹
    parser.add_argument('-s', '--split_dir', type=str, default='./split',
                        help='directory of the split json file')
    # 数据压缩
    parser.add_argument('-c', '--compression', type=str, default='raw',
                        help='compression of video')
    args = parser.parse_args()
    return args


def main(args):
    train_idx, val_idx, test_idx = get_split_index(args.split_dir)

    for dataset_path in DATASET_PATHS.values():
        videos_path = join(args.data_path, dataset_path, args.compression, 'videos')
        images_path = join(args.output_path, dataset_path, args.compression, 'images')
        dataset_frame_count = 0
        dataset_face_miss_count = 0
        for video in tqdm(os.listdir(videos_path)):
            image_folder = video.split('.')[0]
            if os.path.exists(join(images_path, image_folder)):
                continue
            # 根据json文件判断是否是训练数据
            is_train = False
            current_idx = video[:3]
            if str(current_idx) in train_idx:
                is_train = True
            if 'original' in dataset_path:
                frame_count, face_miss_count = extract_frames(join(videos_path, video),
                                                              join(images_path, image_folder),
                                                              compression=args.compression,
                                                              video_type='original',
                                                              is_train=is_train)
            elif 'NeuralTextures' in dataset_path:
                frame_count, face_miss_count = extract_frames(join(videos_path, video),
                                                              join(images_path, image_folder),
                                                              compression=args.compression,
                                                              video_type='NeuralTextures',
                                                              is_train=is_train)
            else:
                frame_count, face_miss_count = extract_frames(join(videos_path, video),
                                                              join(images_path, image_folder),
                                                              compression=args.compression,
                                                              video_type='other',
                                                              is_train=is_train)
            if not is_train:
                dataset_frame_count += frame_count
                dataset_face_miss_count += face_miss_count
        print(f'dataset:{dataset_path} TEST_SET frame:{dataset_frame_count} miss:{dataset_face_miss_count}')
        os.makedirs(join(args.output_path, dataset_path),exist_ok=True)
        out_file = join(args.output_path, dataset_path, 'test_miss.txt')
        with open(out_file, 'w') as f:
            f.write(f'{dataset_frame_count}\n{dataset_face_miss_count}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
