import torch.utils.data as data
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import random
from transform import *
import numpy as np


class Kinetics(data.Dataset):
    def __init__(self, video_path, slice_num, batch_size, frame_num=9, img_size=112, overlap_rate=0.25):
        self.samples = []
        actions = sorted(os.listdir(video_path))

        self.action_emb = {action: actions.index(action) for action in actions}
        for action in actions:
            train_videos = os.listdir(video_path + '/' + action)
            train_videos.sort()

            for video in train_videos:
                self.samples.append(action + '/' + video)

        self.video_path = video_path
        self.frame_num = frame_num
        self.img_size = img_size
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        self.argumentation = transforms.Compose([
            RandomHorizontalFlip(p=0.5),
            #RandomVerticalFlip(p=0.2),
            #RandomGrayscale(p=0.4),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            RandomCenterCrop(size=img_size, p=0.2)
            ])

        self.to_pil_image = transforms.ToPILImage()
        self.slice_num = slice_num
        self.batch_size = batch_size
        self.overlap_rate = overlap_rate

        random.shuffle(self.samples)
        num = len(self.samples)
        if -1 * (num % batch_size) == 0:
            self.samples = self.samples
        else:
            self.samples = self.samples[0: -1 * (num % batch_size)]

        self.samples = np.array(self.samples)
        self.samples = self.samples.reshape(-1, batch_size)

        print("Samples: " + str(self.samples.shape))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_names = self.samples[idx]
        length = self.frame_num
        inputs = torch.zeros(self.slice_num, self.batch_size, length, 3, self.img_size, self.img_size)
        actions = torch.FloatTensor(self.slice_num, self.batch_size, length)
        overlap_frame_num = torch.FloatTensor(self.slice_num, self.batch_size, 1)

        for b, video_name in enumerate(video_names):
            frames_ = os.listdir(self.video_path + '/' + video_name)
            frames = []
            for frame in frames_:
                if '.jpg' in frame:
                    frames.append(frame)
            frames.sort()

            need_frame = float(length * (1 - self.overlap_rate) * (self.slice_num - 1)) + length
            self.space = max(1, int(round(float(len(frames)) / float(need_frame))))
            imgs = []
            v_length = len(frames)
            for i in range(v_length):
                if i % self.space == self.space - 1:
                    try:
                        img = Image.open(self.video_path + '/' + video_name + '/' + frames[i])
                        imgs.append(img)
                    except:
                        pass

            sample = []

            sample_area = max(0, len(imgs) - length)
            stride = min(int(length * (1 - self.overlap_rate)), int(sample_area / self.slice_num))

            for i in range(self.slice_num):
                start_point = stride * i

                for j in range(length):
                    n_frame = start_point + j
                    if n_frame < len(imgs):
                        sample.append(imgs[n_frame])
                try:
                    while len(sample) < length * (i + 1):
                        sample.append(sample[-1])
                except:
                    print(video_name)

            input = torch.zeros(self.slice_num, length, 3, self.img_size, self.img_size)
            sample = self.argumentation(sample)
            for i in range(self.slice_num):
                for j in range(length):
                    img = sample[i * length + j].resize((self.img_size, self.img_size))
                    transformed_img = self.transform(img)
                    img.close()
                    input[i, j, :, :, :] = transformed_img

            action = torch.FloatTensor(self.slice_num, length)
            for batch in range(self.slice_num):
                for fn in range(0, length):
                    action[batch, fn] = self.action_emb[video_name.split('/')[0]]

            inputs[:, b] = input
            actions[:, b] = action

            overlap_frame_num[:, b] = torch.ones(self.slice_num, 1) * (length - stride)

        return inputs, 0, overlap_frame_num, actions
