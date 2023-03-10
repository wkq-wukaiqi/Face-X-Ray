from HRNet import get_net
import torch
from torchvision.transforms import transforms
import cv2
import os
import numpy as np
import argparse
import dlib
from utils import load_checkpoint
from tqdm import tqdm
from torch import nn

dlib_model_path = './dlib_model/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_model_path)


def predict_with_model(image, model, post_function=nn.Softmax(dim=1)):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required

    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real)
    """
    preprocessed_image = transforms.ToTensor()(image).cuda()
    preprocessed_image = preprocessed_image.unsqueeze(0).cuda()

    # Model prediction
    with torch.no_grad():
        mask, output = model(preprocessed_image)
    output = post_function(output)

    # Cast to desired
    _, prediction = torch.max(output, 1)  # argmax
    prediction = float(prediction.cpu().numpy())

    return int(prediction), output, mask


def get_boundingbox(face, width, height, scale=1.7, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def test_full_image_network(video_path, model, output_path, start_frame=0, end_frame=None):
    print('Starting: {}'.format(video_path))

    # Read and write
    reader = cv2.VideoCapture(video_path)

    video_fn = video_path.split('/')[-1].split('.')[0] + '.avi'
    os.makedirs(output_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None

    # Face detector
    face_detector = dlib.get_frontal_face_detector()

    # Text variables
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    # Frame numbers and length of output video
    frame_num = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    pbar = tqdm(total=end_frame - start_frame)

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1

        if frame_num < start_frame:
            continue
        pbar.update(1)

        # Image size
        height, width = image.shape[:2]

        # Init output writer
        if writer is None:
            writer = cv2.VideoWriter(os.path.join(output_path, video_fn), fourcc, fps,
                                     (height, width)[::-1])
        faces = face_detector(image, 0)
        if len(faces):
            # For now only take biggest face
            face = faces[0]

            # --- Prediction ---------------------------------------------------
            # Face crop with dlib and bounding box scale enlargement
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y + size, x:x + size]

            cropped_face = cv2.resize(cropped_face, (256, 256))

            # Actual prediction using our model
            prediction, output, mask = predict_with_model(cropped_face, model)
            mask = (mask.detach().cpu().numpy()[0] * 255).astype(np.uint8)
            mask = np.transpose(mask, (1, 2, 0))
            mask = np.repeat(mask, repeats=3, axis=2)
            # ------------------------------------------------------------------

            # Text and bb
            # x = face.left()
            # y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            label = 'fake' if prediction == 1 else 'real'
            color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
            output_list = ['{0:.2f}'.format(float(x)) for x in
                           output.detach().cpu().numpy()[0]]
            cv2.putText(image, str(output_list) + '=>' + label, (x, y + h + 80),
                        font_face, font_scale,
                        color, thickness, 2)
            # draw box over face
            # cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            blended_img = cv2.addWeighted(cropped_face, 0.5, mask, 0.5, 0)
            blended_img = cv2.resize(blended_img, (size, size))
            image[y:y + size, x:x + size, :] = blended_img

        if frame_num >= end_frame:
            break

        # Show
        cv2.imshow('test', image)
        cv2.waitKey(33)  # About 30 fps
        writer.write(image)
        faces = face_detector(image, 1)
        # 2. Detect with dlib
    pbar.close()
    if writer is not None:
        writer.release()
        print('Finished! Output saved under {}'.format(output_path))
    else:
        print('Input video file was empty')


def main(args):
    assert torch.cuda.is_available()
    devices = [torch.device('cuda:0'), ]
    net = get_net(cfg_file='HRNet/hrnet_config/experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml',
                  devices=devices)

    new_state_dict = load_checkpoint(os.path.join(args.ckpt_dir, 'best_model.pth.tar'), devices)[0]
    net.load_state_dict(new_state_dict)
    net.eval()

    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)

    test_full_image_network(video_path=args.video,
                            model=net,
                            output_path=args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, help='Directory of checkpoint')
    parser.add_argument('-v', '--video', type=str, help='Video to detect')
    parser.add_argument('-o', '--output_dir', type=str, default='./detect_result',
                        help='Directory to output the result')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
