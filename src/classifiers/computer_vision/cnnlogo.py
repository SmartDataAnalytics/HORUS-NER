from __future__ import division

import torch
import torch.nn as nn
import matplotlib.image as mpimg
from torch.autograd import Variable
import cv2

from src.classifiers.computer_vision.places365 import Places365CV
from src.config import HorusConfig


class CNNLogo(nn.Module):
    def __init__(self, config):
        try:
            super(CNNLogo, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=5, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2))
            self.fc1 = nn.Linear(7 * 7 * 32, 100)
            self.fc2 = nn.Linear(100, 10)
            self.config = config

            self.load_state_dict(torch.load(self.config.models_cnn_org))
            self.eval()
        except:
            raise

    def __postprocess_image(self, image):
        try:
            img = torch.from_numpy(image / float(255)).float()
            img.unsqueeze_(-1)
            img = img.expand(28, 28, 3)
            img = img.transpose(2, 0)
            img.unsqueeze_(0)
            return Variable(img)
        except Exception as e:
            self.config.logger.error(e)
            return False

    def preprocess_image(self, img):
        try:
            img = mpimg.imread(img)
            if len(img.shape) == 3:
                r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            img = 0.2989 * r + 0.5870 * g + 0.1140 * b
            resized_image = cv2.resize(img, (28, 28))
            return self.__postprocess_image(resized_image)
        except Exception as e:
            self.config.logger.error(e)
            return False


#    def preprocess_image(self, img):
#
#        img = mpimg.imread(img)
#
#        # if len(img.shape) == 3:
#        #    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
#        #    img = 0.2989 * r + 0.5870 * g + 0.1140 * b
#        resized_image = cv2.resize(img, (28, 28))
#        return resized_image


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def predict(self, image):
        try:
            image = self.preprocess_image(image)
            if image is not False:
                outputs = self(image)
                _, predicted = torch.max(outputs.data, 1)
                return predicted.numpy().sum()
            else:
                raise Exception('pre-processing error')
        except Exception as e:
            self.config.logger.error(e)
            return 0


if __name__ == '__main__':

    config = HorusConfig()
    image = config.cache_img_folder + '172_0_9.jpg'
    classifier = CNNLogo(config)

    import pathlib


    dir = pathlib.Path(config.cache_img_folder + '../temp-tests/not-logo/')
    pat = "*.jpg"
    max=9999999
    i=0
    tot_logo=0

    for file in dir.glob(pat):
        if i>max:
            break
        i+=1
        try:
            y=classifier.predict(file)
            print(file, y)
            if y==1:
                tot_logo+=1
        except:
            raise

    print(tot_logo, i, tot_logo/i)


