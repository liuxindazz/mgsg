import os
import pdb
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from scipy.io import loadmat
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Food500Dataset(Dataset):
    """
    # Description:
        Dataset for retrieving FGVC Aircraft images and labels

    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self, datapath, train=True, transform=None):
        self.train = train
        self.datapath = datapath

        if self.train:
            list_path = os.path.join(self.datapath, 'train_finetune.txt')
        else:
            list_path = os.path.join(self.datapath, 'test_finetune.txt')

        self.images = []
        self.labels = []
        with open(list_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                words = line.split()
                self.images.append(words[0])
                self.labels.append(int(words[1]))

        # transform
        self.transform = transform

    def __getitem__(self, item):
        # image
        image = Image.open(os.path.join(self.datapath, 'images', '%s' % self.images[item])).convert('RGB')  # (C, H, W)
        image = self.transform(image)

        # return image and label
        return image, self.labels[item]  # count begin from zero

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
                        transforms.RandomHorizontalFlip(p=0.5), # default value is 0.5
                        transforms.Resize((448, 448)),
                        transforms.ToTensor(),
                        normalize
                    ])
    ds = Food200Dataset(datapath = 'D://food200', train=False, transform=train_transforms)
    print(len(ds))
    for i in range(0, 100):
        image, label = ds[i]
        print(image.shape, label)
