import torch.utils.data as data
import os
from PIL import Image
import torch
import cv2
import torchvision.transforms as transforms
import random

class Kinetics(data.Dataset):
    def __init__(self, video_path, img_size=368, space=5, split_num=8, lenn=10, num_class=400):
        self.samples = []
        self.split_num = split_num
        self.len = lenn

        actions = sorted(os.listdir(video_path))
        self.action_emb = {action: actions.index(action) for action in actions}
        for action in actions:
            test_videos = os.listdir(video_path + '/' + action)
            test_videos.sort()

            for video in test_videos:
                self.samples.append(action + '/' + video)
        self.video_path = video_path
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        self.to_pil_image = transforms.ToPILImage()
        self.space = space
        random.shuffle(self.samples)
        random.shuffle(self.samples)
        random.shuffle(self.samples)

        print("Samples: " + str(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_name = self.samples[idx]

        frames_ = os.listdir(self.video_path + '/' + video_name)
        frames = []
        for frame in frames_:
            if '.jpg' in frame:
                frames.append(frame)
        frames.sort()

        imgs = []
        v_length = len(frames)
        need_frame = self.len
        self.space = max(1, round(float(v_length) / float(need_frame)))
        for i in range(v_length):
            if i % self.space == self.space - 1:
                try:
                    img = Image.open(self.video_path + '/' + video_name + '/' + frames[i])
                    imgs.append(img)
                except:
                    pass

        length = len(imgs)
        input = torch.zeros(length, 3, self.img_size, self.img_size)
        for i in range(length):
            img = imgs[i].resize((self.img_size, self.img_size), Image.ANTIALIAS)
            transformed_img = self.transform(img)
            img.close()
            input[i, :, :, :] = transformed_img
        action = torch.FloatTensor(length)
        for fn in range(0, length):
            action[fn] = self.action_emb[video_name.split('/')[0]]

        L = self.len
        if len(input) >= L:
            return input[0: L], 0, 0, action[0: L]
        if len(input) < L:
            input_ = torch.zeros(L, 3, self.img_size, self.img_size)
            action_ = torch.FloatTensor(L)
            for i in range(len(input)):
                input_[i] = input[i]
                action_[i] = action[i]
            for i in range(L - len(input)):
                input_[len(input) + i] = input[-1]
                action_[len(input) + i] = action[-1]
            return input_, 0, 0, action_

