import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torch.autograd import Variable

from horus.core.config import HorusConfig


class CNN(nn.Module):
    def __init__(self):
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
        self.config = HorusConfig()



    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def detect_faces(self, image):
        img = torch.from_numpy(image / float(255)).float()
        img.unsqueeze_(-1)
        img = img.expand(28, 28, 3)
        img = img.transpose(2, 0)
        img.unsqueeze_(0)
        img_variable = Variable(img)
        cnn = CNN()
        cnn.load_state_dict(torch.load(self.config.models_cnn_per))
        cnn.eval()
        outputs = cnn(img_variable)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy().sum()

    def detect_logo_cnn(self, image):
        img = torch.from_numpy(image / float(255)).float()
        img.unsqueeze_(-1)
        img = img.expand(28, 28, 3)
        img = img.transpose(2, 0)
        img.unsqueeze_(0)
        img_variable = Variable(img)
        cnn = CNN()
        cnn.load_state_dict(torch.load(self.config.models_cnn_org))
        cnn.eval()
        outputs = cnn(img_variable)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy().sum()

    def detect_place_cnn(self, image):
        ret = []
        img = torch.from_numpy(image / float(255)).float()
        img.unsqueeze_(-1)
        img = img.expand(28, 28, 3)
        img = img.transpose(2, 0)
        img.unsqueeze_(0)
        img_variable = Variable(img)
        cnn = CNN()
        # loc1
        cnn.load_state_dict(torch.load(self.config.models_cnn_loc1))
        cnn.eval()
        outputs = cnn(img_variable)
        _, predicted = torch.max(outputs.data, 1)
        ret.append(predicted.numpy().sum())
        # loc2
        cnn.load_state_dict(torch.load(self.config.models_cnn_loc2))
        cnn.eval()
        outputs = cnn(img_variable)
        _, predicted = torch.max(outputs.data, 1)
        ret.append(predicted.numpy().sum())
        # loc3
        cnn.load_state_dict(torch.load(self.config.models_cnn_loc3))
        cnn.eval()
        outputs = cnn(img_variable)
        _, predicted = torch.max(outputs.data, 1)
        ret.append(predicted.numpy().sum())
        # loc4
        cnn.load_state_dict(torch.load(self.config.models_cnn_loc4))
        cnn.eval()
        outputs = cnn(img_variable)
        _, predicted = torch.max(outputs.data, 1)
        ret.append(predicted.numpy().sum())
        # loc5
        cnn.load_state_dict(torch.load(self.config.models_cnn_loc5))
        cnn.eval()
        outputs = cnn(img_variable)
        _, predicted = torch.max(outputs.data, 1)
        ret.append(predicted.numpy().sum())
        # loc6
        cnn.load_state_dict(torch.load(self.config.models_cnn_loc6))
        cnn.eval()
        outputs = cnn(img_variable)
        _, predicted = torch.max(outputs.data, 1)
        ret.append(predicted.numpy().sum())
        # loc7
        cnn.load_state_dict(torch.load(self.config.models_cnn_loc7))
        cnn.eval()
        outputs = cnn(img_variable)
        _, predicted = torch.max(outputs.data, 1)
        ret.append(predicted.numpy().sum())
        # loc8
        cnn.load_state_dict(torch.load(self.config.models_cnn_loc8))
        cnn.eval()
        outputs = cnn(img_variable)
        _, predicted = torch.max(outputs.data, 1)
        ret.append(predicted.numpy().sum())
        # loc9
        cnn.load_state_dict(torch.load(self.config.models_cnn_loc9))
        cnn.eval()
        outputs = cnn(img_variable)
        _, predicted = torch.max(outputs.data, 1)
        ret.append(predicted.numpy().sum())
        # loc10
        cnn.load_state_dict(torch.load(self.config.models_cnn_loc10))
        cnn.eval()
        outputs = cnn(img_variable)
        _, predicted = torch.max(outputs.data, 1)
        ret.append(predicted.numpy().sum())
        return ret