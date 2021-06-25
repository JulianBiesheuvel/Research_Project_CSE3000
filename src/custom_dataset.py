import torch
import csv
from numpy.random import randint
from skimage.util import random_noise
from torch.utils.data import Dataset
from torchvision.transforms import RandomApply
from torchvision.transforms.functional import rotate, invert


class CustomMNISTDataset(Dataset):

    def __init__(self, images, transform=None, apply_transform=False, experiment=None):
        self.images = images
        self.transform = transform
        self.len = images.shape[0]
        self.apply_transform = apply_transform
        self.experiment = experiment

    def __getitem__(self, index):

        image = self.images[index].unsqueeze(0)
        # Regular
        # img_trans = self.images[index].unsqueeze(0)
        # Inverted
        img_trans = invert(image)
        # Just Rotated
        # img_trans = rotate(image, angle=45, fill=[0, ])
        # Random Rotation
        # deg = randint(360)
        #
        # with open('../vis3/distribution_' + self.experiment + '_train_hist_randrot.csv', mode='a', newline="\n") as dataset:
        #     image_writer = csv.writer(dataset, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #     image_writer.writerow([deg])

        # img_trans = rotate(image, angle=deg, fill=[0, ])


        #==============NOISY2====================
        #==============MEAN,STD,VAR==============
        # x = torch.zeros(28, 28, dtype=torch.float64)
        # x = x + torch.randn(x.size()) * torch.std(image.float()) + torch.mean(image.float())

        #=============NOISY2=====================
        #============MEDIAN======================

        # y = torch.zeros(393)
        #
        # val = randint(50, 230)
        #
        # x = torch.distributions.uniform.Uniform(0, val).sample([1, 391])
        #
        # r = torch.cat((x[0], y))
        #
        # r = r[torch.randperm(r.size()[0])]
        #
        # r = torch.reshape(r, (28, 28))
        #
        # r = r[torch.randperm(r.size()[0])]
        # r = r[:, torch.randperm(r.size()[1])]
        #
        # r = r.unsqueeze(0)

        switcher = {
            'mean': [torch.mean(img_trans.float()), torch.mean(image.float())],
            'variance': [torch.var(img_trans.float()), torch.var(image.float())],
            'std': [torch.std(img_trans.float()), torch.std(image.float())],
            'median': [torch.median(img_trans.float()), torch.median(image.float())]
        }

        # FOR MEDIAN IT SHOULD BE R AND OTHERS X
        # switcher_noisy = {
        #     'mean': [torch.mean(x)],
        #     'variance': [torch.var(x)],
        #     'std': [torch.std(x)],
        #     'median': [torch.median(x)]
        # }

        if self.apply_transform:
            return img_trans, switcher.get(self.experiment)[0]
        else:
            # return x, switcher_noisy.get(self.experiment)[0]
            # return r, switcher_noisy.get(self.experiment)[0]
            return image, switcher.get(self.experiment)[1]

    def __len__(self):
        return self.len

