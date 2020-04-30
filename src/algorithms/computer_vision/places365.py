import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image

from config import HorusConfig


class Places365CV():
    def __init__(self, config):
        try:
            self.config = config
            self.arch = 'resnet18'
            # load the pre-trained weights
            self.model_file = 'whole_%s_places365.pth.tar' % self.arch
            #if not os.access(self.model_file, os.W_OK):
            #    weight_url = 'http://places2.csail.mit.edu/models_places365/' + self.model_file
            #    os.system('wget ' + weight_url)

            useGPU = 0
            if useGPU == 1:
                self.model = torch.load(config.dir_models + "/places365/" + self.model_file)
            else:
                self.model = torch.load(config.dir_models + "/places365/" + self.model_file,
                                   map_location=lambda storage, loc: storage) # model trained in GPU could be deployed in CPU machine like this!

            ## assume all the script in python36, so the following is not necessary
            ## if you encounter the UnicodeDecodeError when use python3 to load the model, add the following line will fix it. Thanks to @soravux
            #from functools import partial
            #import pickle
            #pickle.load = partial(pickle.load, encoding="latin1")
            #pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
            #model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)
            #torch.save(model, 'whole_%s_places365_python36.pth.tar'%arch)

            self.model.eval()

            # load the image transformer
            self.centre_crop = trn.Compose([
                    trn.Resize((256,256)),
                    trn.CenterCrop(224),
                    trn.ToTensor(),
                    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            # load the class label
            file_name = 'categories_places365.txt'
            #if not os.access(file_name, os.W_OK):
            #    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            #    os.system('wget ' + synset_url)
            self.classes = list()
            with open(config.dir_models + "/places365/" + file_name) as class_file:
                for line in class_file:
                    self.classes.append(line.strip().split(' ')[0][3:])
            self.classes = tuple(self.classes)
        except Exception as e:
            raise e

    def predict(self, img_name, top=5):
        try:
            img = Image.open(img_name)
            input_img = V(self.centre_crop(img).unsqueeze(0), volatile=True)
            # forward pass
            logit = self.model.forward(input_img)
            h_x = F.softmax(logit).data.squeeze()
            probs, idx = h_x.sort(0, True)
            out = []
            for i in range(0, top):
                out.append((self.classes[idx[i]], probs[i]))
                #print('{:.3f} -> {}'.format(probs[i], self.classes[idx[i]]))
            return out

        except Exception as e:
            raise e


if __name__ == '__main__':

    config = HorusConfig()
    placesCNN = Places365CV(config)
    print(placesCNN.predict(config.cache_img_folder + '172_0_9.jpg'))