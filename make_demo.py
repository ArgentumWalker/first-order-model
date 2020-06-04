import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import cv2

import imageio
import numpy as np
from skimage.transform import resize
from collections import deque
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull


def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector


class Animator:
    def __init__(self, target_images, start_image, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
        self.cpu = cpu
        self.kp_detector = kp_detector
        self.generator = generator
        self.relative = relative
        self.adapt_movement_scale = adapt_movement_scale

        self.targets, self.kp_targets = [], []
        for img in target_images:
            tgt, kp = self._tensor_image(img)
            self.targets.append(tgt)
            self.kp_targets.append(kp)
        _, self.kp_start = self._tensor_image(start_image)

    def _tensor_image(self, image):
        with torch.no_grad():
            image = torch.tensor(image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            if not self.cpu:
                image = image.cuda()
            return image, self.kp_detector(image)

    def next(self, image):
        _, kp_image = self._tensor_image(image)
        result = []
        for tgt, kp in zip(self.targets, self.kp_targets):
            with torch.no_grad():
                norm = normalize_kp(kp, kp_image, self.kp_start, self.adapt_movement_scale, self.relative, self.relative)
                out = self.generator(tgt, kp_source=kp, kp_driving=norm)
            result.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
        return result

    def reset(self, image):
        _, self.kp_start = self._tensor_image(image)


if __name__ == "__main__":
    generator, kp_detector = load_checkpoints(config_path='config/vox-adv-256.yaml',
                                              checkpoint_path='./vox-adv-cpk.pth.tar')

    target_images = [
        resize(imageio.imread('face_on/ded1.jpg'), (256, 256)),
        resize(imageio.imread('face_on/Viking.jpg'), (256, 256)),
        resize(imageio.imread('face_on/Girl2.jpg'), (256, 256)),
        resize(imageio.imread('face_on/woman2.jpg'), (256, 256))
    ]
    #target_images = [
    #    resize(imageio.imread('face_on/Bowie_2.jpg'), (256, 256))
    #]
    crop_y = 250
    crop_x = 00
    start_img = np.zeros_like(target_images[0])
    pred_img = np.zeros_like(target_images[0])

    for k in range(13):
        cap = cv2.VideoCapture(f'face_on/video_v_{k}.mp4')
        wr = imageio.get_writer(f'face_on/result_v_{k}.mp4', fps=30)
        #wr = cv2.VideoWriter('face_on/result.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10,
        #                     (target_images[0].shape[1] * (len(target_images) + 1), target_images[0].shape[0]), True)
        animator = None
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()

            if frame is None:
                break

            #if i > 30:
            #    break

            #frame = frame[:, crop_const:crop_const+360]
            if k < 2:
                frame = frame[100:100+360, :]
            else:
                frame = frame[300:300 + 1080,:]

            frame = resize(frame[:, :, ::-1], (256, 256), order=0)
            if i == 0:
                animator = Animator(target_images, frame, generator, kp_detector)
                start_img = frame
            i += 1
            pred_frames = animator.next(frame)
            res = np.concatenate(list(pred_frames) + [frame], axis=1)[:, :, ::-1]
            #wr.write(res)
            wr.append_data(res[:, :, ::-1])
            cv2.imshow("Frame", res)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            print(i)

        #wr.release()
        wr.close()