#!/usr/bin/python
from torchvision.datasets import STL10
import utils as ut

class STL10(STL10):
    def grayscale(self) :
        #TODO
        return self

    def __getitem__(self, index):
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        gray = self.grayscale(img)

        return img, gray
