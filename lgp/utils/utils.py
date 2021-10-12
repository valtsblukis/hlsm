import cv2
import os
import numpy as np
import functools
from imageio import imwrite
import moviepy.editor as mpy
from lgp import paths
from lgp.parameters import Hyperparams


def save_png(frame, fname):
    #frame = standardize_image(frame, normalize=False)
    imwrite(fname, frame)


def save_frames(frames, framedir, start_t=0):
    os.makedirs(framedir, exist_ok=True)
    for i, frame in enumerate(frames):
        fname = os.path.join(framedir, f"frame_{start_t + i:05d}.jpg")
        save_png(frame, fname)


def save_gif(frames, gif_name, fps: float = 2):
    if len(frames) > 0 and frames[0].max() < 1.001 and not frames[0].dtype == np.uint8:
        frames = [(f * 255).astype(np.uint8) for f in frames]
    clip = mpy.ImageSequenceClip(frames, fps=fps)
    try:
        clip.write_gif(gif_name)
    except Exception as e:
        print(f"ERROR: Failed to write gif: {gif_name}")


def save_mp4(frames, mp4_name, fps: float = 1.0):
    if len(frames) > 0 and frames[0].max() < 1.001 and not frames[0].dtype == np.uint8:
        frames = [(f * 255).astype(np.uint8) for f in frames]
    clip = mpy.ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(mp4_name, fps=fps)


def tensorshow(name, ndarray_or_tensor, scale=1, normalize=True, waitkey=1):
    img = standardize_image(ndarray_or_tensor, scale, normalize)

    # Convert to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imshow(name, img)
    cv2.waitKey(waitkey)


def tensorsave(name, ndarray_or_tensor, scale=1, normalize=True, waitkey=1, idx=0):
    img = standardize_image(ndarray_or_tensor, scale, normalize)
    tensorsaves_dir = os.path.join(paths.get_artifact_output_path(), "tensorsaves")
    fname = f"{tensorsaves_dir}/{name}_{idx}.png"
    os.makedirs(tensorsaves_dir, exist_ok=True)
    imsave(fname, img)


def standardize_image(ndarray_or_tensor, scale=1, normalize=True, uint8=False):
    if isinstance(ndarray_or_tensor, torch.Tensor):
        img = ndarray_or_tensor.detach().cpu().numpy()
        if len(img.shape) == 4:
            assert img.shape[0] == 1
            img = img[0]
        if len(img.shape) == 3:
            img = img.transpose((1, 2, 0))
    else:
        img = ndarray_or_tensor

    # Convert to float
    img = img.astype(np.float32)

    # If no channel dimension (this is a grayscale image), add one
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]

    # If >3 channels, show the first 3:
    if img.shape[2] > 3:
        img = img[:, :, :3]

    # If ==2 channels, add another one to make it compatible with cv2.imshow
    if img.shape[2] == 2:
        img = np.concatenate([img, np.zeros_like(img[:, :, 0:1])], axis=2)
    # If ==1 channels, add the color channel for consistency
    if img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=2)

    # Shift and scale to 0-1 range
    if normalize:
        img -= img.min()
        img /= (img.max() + 1e-10)

    # Scale according to the scaling factor
    if scale != 1:
        img = cv2.resize(img,
                         dsize=(int(img.shape[1] * scale), int(img.shape[0] * scale)),
                         interpolation=cv2.INTER_NEAREST)

    if uint8:
        if np.max(img) < 1.01:
            img = img * 255
        img = img.astype(np.uint8)

    return img


import pprint

import torch


def millis():
    import datetime
    stampnow = datetime.datetime.now().timestamp()
    m = stampnow * 1000.0
    return m

"""
A profiler used to time execution of code.
Every time "tick" is called, it adds the amount of elapsed time to the "key" accumulator
This allows timing multiple things simultaneously and keeping track of how much time each takes
"""
class SimpleProfilerDummy():
    def __init__(self, torch_sync=False, print=True):
        pass

    def reset(self):
        pass

    def tick(self, key):
        pass

    def loop(self):
        pass

    def print_stats(self, every_n_times=1):
        pass

class SimpleProfilerReal():
    def __init__(self, torch_sync=False, print=True):
        """
        When debugging GPU code, torch_sync must be true, because GPU and CPU computation is asynchronous
        :param torch_sync: If true, will call cuda synchronize.
        :param print: If true, will print stats when print_stats is called. Pass False to disable output for release code
        """
        self.time = millis()
        self.loops = 0
        self.times = {}
        self.avg_times = {}
        self.sync = torch_sync
        self.print = print
        self.print_time = 0

    def reset(self):
        self.time = millis()
        self.times = {}

    def tick(self, key):
        if key not in self.times:
            self.times[key] = 0

        if self.sync:
            torch.cuda.synchronize()

        now = millis()
        self.times[key] += now - self.time
        self.time = now

    def loop(self):
        self.loops += 1
        for key, time in self.times.items():
            self.avg_times[key] = self.times[key] / self.loops

    def print_stats(self, every_n_times=1):
        self.print_time += 1
        if self.print and self.print_time % every_n_times == 0:
            total_time = 0
            if len(self.avg_times) > 0:
                print("Avg times per loop: ")
                pprint.pprint(self.avg_times)
                for k,v in self.avg_times.items():
                    if k != "out" and k != ".":
                        total_time += v
                print(f"Total avg loop time: {total_time}")
            else:
                print("Cumulative times: ")
                pprint.pprint(self.times)

SimpleProfiler = SimpleProfilerReal