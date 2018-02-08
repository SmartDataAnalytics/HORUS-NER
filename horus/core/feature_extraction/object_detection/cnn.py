import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.image as mpimg
from torch.autograd import Variable
import cv2
from horus.core.config import HorusConfig
from horus.core.util.systemlog import SysLogger


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
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

    def __postprocess_image(self, image):
        try:
            img = torch.from_numpy(image / float(255)).float()
            img.unsqueeze_(-1)
            img = img.expand(28, 28, 3)
            img = img.transpose(2, 0)
            img.unsqueeze_(0)
            return Variable(img)
        except Exception as e:
            raise(e)

    def preprocess_image(self, img):
        try:
            img = mpimg.imread(img)
            if len(img.shape) == 3:
                r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            img = 0.2989 * r + 0.5870 * g + 0.1140 * b
            resized_image = cv2.resize(img, (28, 28))
            return self.__postprocess_image(resized_image)
        except Exception as e:
            raise(e)


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

    def detect_faces(self, image):
        try:
            self.load_state_dict(torch.load(self.config.models_cnn_per))
            self.eval()
            outputs = self(image)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.numpy().sum()
        except:
            return 0

    def detect_logo_cnn(self, image):
        try:
            self.load_state_dict(torch.load(self.config.models_cnn_org))
            self.eval()
            outputs = self(image)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.numpy().sum()
        except:
            return 0

    def detect_place_cnn(self, image):
        try:
            ret = []
            # loc1
            self.load_state_dict(torch.load(self.config.models_cnn_loc1))
            self.eval()
            outputs = self(image)
            _, predicted = torch.max(outputs.data, 1)
            ret.append(predicted.numpy().sum())
            # loc2
            self.load_state_dict(torch.load(self.config.models_cnn_loc2))
            self.eval()
            outputs = self(image)
            _, predicted = torch.max(outputs.data, 1)
            ret.append(predicted.numpy().sum())
            # loc3
            self.load_state_dict(torch.load(self.config.models_cnn_loc3))
            self.eval()
            outputs = self(image)
            _, predicted = torch.max(outputs.data, 1)
            ret.append(predicted.numpy().sum())
            # loc4
            self.load_state_dict(torch.load(self.config.models_cnn_loc4))
            self.eval()
            outputs = self(image)
            _, predicted = torch.max(outputs.data, 1)
            ret.append(predicted.numpy().sum())
            # loc5
            self.load_state_dict(torch.load(self.config.models_cnn_loc5))
            self.eval()
            outputs = self(image)
            _, predicted = torch.max(outputs.data, 1)
            ret.append(predicted.numpy().sum())
            # loc6
            self.load_state_dict(torch.load(self.config.models_cnn_loc6))
            self.eval()
            outputs = self(image)
            _, predicted = torch.max(outputs.data, 1)
            ret.append(predicted.numpy().sum())
            # loc7
            self.load_state_dict(torch.load(self.config.models_cnn_loc7))
            self.eval()
            outputs = self(image)
            _, predicted = torch.max(outputs.data, 1)
            ret.append(predicted.numpy().sum())
            # loc8
            self.load_state_dict(torch.load(self.config.models_cnn_loc8))
            self.eval()
            outputs = self(image)
            _, predicted = torch.max(outputs.data, 1)
            ret.append(predicted.numpy().sum())
            # loc9
            self.load_state_dict(torch.load(self.config.models_cnn_loc9))
            self.eval()
            outputs = self(image)
            _, predicted = torch.max(outputs.data, 1)
            ret.append(predicted.numpy().sum())
            # loc10
            self.load_state_dict(torch.load(self.config.models_cnn_loc10))
            self.eval()
            outputs = self(image)
            _, predicted = torch.max(outputs.data, 1)
            ret.append(predicted.numpy().sum())
            return ret
        except:
            return [0] * 10