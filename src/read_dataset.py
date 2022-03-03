from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch
import numpy as np
import os
from torchvision.io import read_image, ImageReadMode
import cv2


class BrainDataset(Dataset):

    def __init__(self, img_dir, perm=None):
        self.img_dir = img_dir
        self.file_names = os.listdir(img_dir)
        self.vocab_size = 256
        self.block_size = 32 * 32 - 1
        self.perm = torch.arange(32 * 32) if perm is None else perm

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.file_names[idx])
        # pytorch image reading
        image = read_image(img_path, ImageReadMode.GRAY)
        # flatten out all pixels
        flattened = torch.from_numpy(np.array(image)).view(-1)

        """
        # same approach with OpenCV image reading
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # flatten out all pixels
        flattened = torch.from_numpy(image).view(-1)
        """

        # permutation
        flattened = flattened[self.perm]
        flattened = flattened.long()
        return flattened[:-1], flattened[1:]









# print("brain_dataset[0]:   ")
# print(self.brain_dataset[0])
# print(len(self.brain_dataset[0]))

# x,y = self.brain_dataset[0]
# x = torch.from_numpy(np.array(x)).view(-1, 1)
# x = x.int()
# data_tensor = torch.from_numpy(x)
# print("sata tensor: ")
# print(data_tensor)
# print("x:")
# print(x[0])
# print("y:")
# print(y)

# z = x[:-1]
# zz = x[1:]
# print("x[:-1]: ")
# print(z)
# print("x[1:]: ")
# print(zz)