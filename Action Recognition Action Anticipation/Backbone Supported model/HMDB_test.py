import torch.utils.data as data
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import random
from transform import *



class HMDB(data.Dataset):
    def __init__(self, video_path, frame_num, action_num=51, img_size=368, anticipation=False):
        self.samples = []
        actions = sorted(os.listdir(video_path+'/' + 'test'))

        self.action_emb = {action: actions.index(action) for action in actions}

        for action in actions:
            test_videos = os.listdir(video_path + '/' + 'test' + '/' + action)
            test_videos.sort()
            for video in test_videos:
                self.samples.append('test' + '/' + action + '/' + video)
        self.video_path = video_path
        self.action_num = action_num
        self.img_size = img_size
        self.frame_num = frame_num
        self.anticipation = anticipation
        self.transform = transforms.Compose([  # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        self.transform2 = transforms.Compose([
            RandomHorizontalFlip(p=0.2),
            RandomVerticalFlip(p=0.2),
            RandomGrayscale(p=0.2),
            #ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            RandomCenterCrop(size=img_size, p=0.3)])


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
        if self.anticipation:
            frames = frames[0:2]

        need_frame = self.frame_num
        self.space = max(1, int(round(float(len(frames)) / float(need_frame))))
        imgs = []
        length = len(frames)
        for i in range(length):
            if i % self.space == self.space - 1:
                img = Image.open(self.video_path + '/' + video_name + '/' + frames[i])
                imgs.append(img)

        while len(imgs) < need_frame:
            imgs.append(imgs[-1])

        input = torch.zeros(need_frame, 3, self.img_size, self.img_size)
        for i in range(need_frame):
            img = imgs[i].resize((self.img_size, self.img_size), Image.ANTIALIAS)
            transformed_img = self.transform(img)
            img.close()
            input[i, :, :, :] = transformed_img
        action = torch.FloatTensor(need_frame)
        for fn in range(0, need_frame):
            action[fn] = self.action_emb[video_name.split('/')[1]]
        return input, 0, 0, action
