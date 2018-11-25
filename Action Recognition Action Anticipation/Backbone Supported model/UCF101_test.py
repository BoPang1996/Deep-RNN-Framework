import torch.utils.data as data
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import random
from transform import *


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
    def __init__(self, video_path, frame_num, action_num=101, img_size=368, anticipation=False):
        self.samples = []
        actions = sorted(os.listdir(video_path+'/' + 'test'))

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
            action[fn] = action_emb[video_name.split('/')[1]]
        return input, 0, 0, action
