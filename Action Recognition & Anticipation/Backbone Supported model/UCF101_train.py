import torch.utils.data as data
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import random
from transform import *
import numpy as np


action_emb = {'ApplyEyeMakeup': 0,'ApplyLipstick': 1,'Archery': 2,'BabyCrawling': 3,'BalanceBeam': 4,
   'BandMarching': 5,'BaseballPitch': 6,'Basketball': 7,'BasketballDunk': 8,'BenchPress': 9,
    'Biking': 10,'Billiards': 11,'BlowDryHair': 12,'BlowingCandles': 13,'BodyWeightSquats': 14,
    'Bowling': 15,'BoxingPunchingBag': 16,'BoxingSpeedBag': 17,'BreastStroke': 18,'BrushingTeeth': 19,
    'CleanAndJerk': 20,'CliffDiving': 21,'CricketBowling': 22,'CricketShot': 23,'CuttingInKitchen': 24,
    'Diving': 25,'Drumming': 26,'Fencing': 27,'FieldHockeyPenalty': 28,'FloorGymnastics': 29,
    'FrisbeeCatch': 30,'FrontCrawl': 31,'GolfSwing': 32,'Haircut': 33,'HammerThrow': 34,
    'Hammering': 35,'HandstandPushups': 36,'HandstandWalking': 37,
    'HeadMassage': 38,'HighJump': 39,'HorseRace': 40,'HorseRiding': 41,'HulaHoop': 42,
    'IceDancing': 43,'JavelinThrow': 44,'JugglingBalls': 45,'JumpRope': 46,'JumpingJack': 47,
    'Kayaking': 48,'Knitting': 49,'LongJump': 50,'Lunges': 51,'MilitaryParade': 52,'Mixing': 53,
    'MoppingFloor': 54,'Nunchucks': 55,'ParallelBars': 56,'PizzaTossing': 57,'PlayingCello': 58,
    'PlayingDaf': 59,'PlayingDhol': 60,'PlayingFlute': 61,'PlayingGuitar': 62,'PlayingPiano': 63,
    'PlayingSitar': 64,'PlayingTabla': 65,'PlayingViolin': 66,'PoleVault': 67,'PommelHorse': 68,
    'PullUps': 69,'Punch': 70,'PushUps': 71,'Rafting': 72,'RockClimbingIndoor': 73,'RopeClimbing': 74,
    'Rowing': 75,'SalsaSpin': 76,'ShavingBeard': 77,'Shotput': 78,'SkateBoarding': 79,'Skiing': 80,
    'Skijet': 81,'SkyDiving': 82,'SoccerJuggling': 83,'SoccerPenalty': 84,'StillRings': 85,'SumoWrestling': 86,
    'Surfing': 87,'Swing': 88,'TableTennisShot': 89,'TaiChi': 90,'TennisSwing': 91,'ThrowDiscus': 92,
    'TrampolineJumping': 93,'Typing': 94,'UnevenBars': 95,'VolleyballSpiking': 96,'WalkingWithDog': 97,
    'WallPushups': 98,'WritingOnBoard': 99,'YoYo' : 100}

class UCF101(data.Dataset):
    def __init__(self, video_path, slice_num, batch_size, frame_num=9, action_num=101, img_size=368, overlap_rate=0.25):
        self.samples = []
        actions = os.listdir(video_path+'/' + 'train')

        for action in actions:
            train_videos = os.listdir(video_path + '/' + 'train' + '/' + action)
            train_videos.sort()
            for video in train_videos:
                self.samples.append('train' + '/' + action + '/' + video)

        self.transform = transforms.Compose([  # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        self.argumentation = transforms.Compose([
            RandomHorizontalFlip(p=0.5),
            RandomGrayscale(p=0.3),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.5),
            RandomCenterCrop(size=img_size, p=0.3)])

        random.shuffle(self.samples)

        num = len(self.samples)
        if -1 * (num % batch_size) == 0:
            self.samples = self.samples
        else:
            self.samples = self.samples[0: -1 * (num % batch_size)]

        self.samples = np.array(self.samples)
        self.samples = self.samples.reshape(-1,batch_size)

        print("Samples: " + str(self.samples.shape))

        self.to_pil_image = transforms.ToPILImage()
        self.slice_num = slice_num
        self.batch_size = batch_size
        self.video_path = video_path
        self.frame_num = frame_num
        self.action_num = action_num
        self.img_size = img_size
        self.overlap_rate = overlap_rate

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

            need_frame = float(length * (1 - self.overlap_rate) * (self.slice_num - 1) + length)
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

                while len(sample) < length * (i + 1):
                    sample.append(sample[-1])


            input = torch.zeros(self.slice_num, length, 3, self.img_size, self.img_size)
            sample = self.argumentation(sample)
            for i in range(self.slice_num):
                for j in range(length):
                    img = sample[i * length + j].resize((self.img_size, self.img_size), Image.ANTIALIAS)
                    transformed_img = self.transform(img)
                    img.close()
                    input[i, j, :, :, :] = transformed_img

            action = torch.FloatTensor(self.slice_num, length)
            for batch in range(self.slice_num):
                for fn in range(0, length):
                    action[batch, fn] = action_emb[video_name.split('/')[1]]

            inputs[:, b] = input
            actions[:,b] = action

            overlap_frame_num[:, b] = torch.ones(self.slice_num, 1) * (length - stride)

        return inputs, 0, overlap_frame_num, actions
