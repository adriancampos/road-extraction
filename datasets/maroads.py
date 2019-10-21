import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision.datasets.folder import is_image_file, default_loader


classes = ['Road']

class_weight = torch.FloatTensor([1])


mean = [0.41189489566336, 0.4251328133025, 0.4326707089857]
std = [0.27413549931506, 0.28506257482912, 0.28284674400252]

class_color = [
    (255, 255, 255),
    (0, 0, 0),
]


def _make_dataset(dir, slc):
    images = []
    masks = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                
                if   os.path.split(path)[0][-3:] == "sat":
                    images.append(path)
                elif os.path.split(path)[0][-3:] == "map":
                    masks.append(path)
                else:
                    raise FileNotFoundError
    
#     if self.split != "valid":
#         assert(len(images) == len(masks))  # sloppy sanity check
#     print("len(images)",len(images))
#     print("len(masks) ",len(masks))

    
    return sorted(images)[slc], sorted(masks)[slc]
#     return sorted(images), sorted(masks)



class LabelToLongTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            label = torch.from_numpy(pic).int()
        else:
            label = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            label = label.view(pic.size[1], pic.size[0], 3)
#             print(label.size())
#             label = label.split(1,0)[0]  # turn rgb into binary
            label = label.transpose(0, 1).transpose(0, 2).squeeze().contiguous().int() / 255
#             print(label.size())
#             label = label[0].unsqueeze(0)
            label = label[0]
#             print(label.size())
#             print(label)
        return label


class LabelTensorToPILImage(object):
    def __call__(self, label):
        label = label.unsqueeze(0)
        colored_label = torch.zeros(3, label.size(1), label.size(2)).byte()
        for i, color in enumerate(class_color):
            mask = label.eq(i)
            for j in range(3):
                colored_label[j].masked_fill_(mask, color[j])
        npimg = colored_label.numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        mode = None
        if npimg.shape[2] == 1:
            npimg = npimg[:, :, 0]
            mode = "L"

        return Image.fromarray(npimg, mode=mode)


class MARoads(data.Dataset):

    def __init__(self, root, joint_transform=None,
                 transform=None, target_transform=LabelToLongTensor(),
                 download=False,
                 loader=default_loader,
                 slc=slice(None)):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.loader = loader
        self.class_weight = class_weight
        self.classes = classes
        self.mean = mean
        self.std = std
        self.slc = slc

        if download:
            self.download()

        self.imgs, self.msks = _make_dataset(self.root, slc)

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
            
        if len(self.msks):
            target = self.loader(self.msks[index])

            if self.joint_transform is not None:
                img, target = self.joint_transform([img, target])

            if self.transform is not None:
                img = self.transform(img)

            target = self.target_transform(target)
            return img, target
        else:
            # If we don't have masks, simply return the image and its path instead
            # TODO: Shouldn't be returning different types depending on input. Fix this.
            if self.transform is not None:
                img = self.transform(img)
            
            return img, path
            

    def __len__(self):
#         return len(self.imgs)
        # for debug
        return 20


    def download(self):
        # TODO: please download the dataset from
        # https://competitions.codalab.org/competitions/18467
        raise NotImplementedError
