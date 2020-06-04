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
    def __init__(self, target_image, start_image, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
        self.cpu = cpu
        self.kp_detector = kp_detector
        self.generator = generator
        self.relative = relative
        self.adapt_movement_scale = adapt_movement_scale

        self.target, self.kp_target = self._tensor_image(target_image)
        _, self.kp_start = self._tensor_image(start_image)

    def _tensor_image(self, image):
        with torch.no_grad():
            image = torch.tensor(image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            if not self.cpu:
                image = image.cuda()
            return image, self.kp_detector(image)

    def next(self, image):
        _, kp_image = self._tensor_image(image)
        with torch.no_grad():
            norm = normalize_kp(self.kp_target, kp_image, self.kp_start, self.adapt_movement_scale, self.relative, self.relative)
            out = self.generator(self.target, kp_source=self.kp_target, kp_driving=norm)
        return np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]

    def reset(self, image):
        _, self.kp_start = self._tensor_image(image)


if __name__ == "__main__":
    generator, kp_detector = load_checkpoints(config_path='config/vox-256-ft.yaml',
                                              checkpoint_path='./vox-ft.pth.tar')

    target_img = resize(imageio.imread('./putin3.jpg'), (256, 256))
    #target_img = resize(imageio.imread('./statue-01.jpg'), (256, 256))
    #target_img = resize(imageio.imread('./mandarin.jpg'), (256, 256))
    #target_img = resize(imageio.imread('./kyzya.jpg'), (256, 256))
    #target_img = resize(imageio.imread('./smithuber.jpg'), (256, 256))
    #target_img = resize(imageio.imread('./oleg2.jpg'), (256, 256))
    start_img = np.zeros_like(target_img)
    pred_img = np.zeros_like(target_img)

    cap = cv2.VideoCapture(0)
    animator = None
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if frame.shape[0] > frame.shape[1]:
            frame = frame[(frame.shape[0] - frame.shape[1])//2:(frame.shape[0] + frame.shape[1])//2]
        if frame.shape[1] > frame.shape[0]:
            frame = frame[:, (frame.shape[1] - frame.shape[0])//2:(frame.shape[1] + frame.shape[0])//2]
        #frame = cv2.fastNlMeansDenoisingColored(frame, templateWindowSize=3, searchWindowSize=5)
        frame = resize(frame[:, :, ::-1], (256, 256), order=0)
        pred_frame = np.zeros_like(frame)
        if i == 10:
            animator = Animator(target_img, frame, generator, kp_detector)
            start_img = frame
        elif i > 10:
            pred_frame = animator.next(frame)
        frame_diff = np.abs(frame - start_img)
        #cv2.imshow('frame', np.concatenate((pred_frame, frame, frame_diff, np.abs(frame - pred_img)), axis=1)[:, :, ::-1])
        cv2.imshow('frame', pred_frame[:, :, ::-1])
        pred_img = frame
        i += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            print("reset", flush=True)
            animator.reset(frame)
            start_img = frame
        if key == ord('q'):
            break
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break