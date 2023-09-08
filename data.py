import os
from collections import Counter

import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


class Plantnet(ImageFolder):
    def __init__(self, root, split, **kwargs):
        self.root = root
        self.split = split
        super().__init__(self.split_folder, **kwargs)

    @property
    def split_folder(self):
        return os.path.join(self.root, self.split)


def get_plantnet_data(root, image_size, crop_size, batch_size, num_workers, pretrained):
    if pretrained:
        transform_train = transforms.Compose([transforms.Resize(size=image_size), transforms.RandomCrop(size=crop_size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                          std=[0.229, 0.224, 0.225])])
        transform_test = transforms.Compose([transforms.Resize(size=image_size), transforms.CenterCrop(size=crop_size),
                                             transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                         std=[0.229, 0.224, 0.225])])
    else:
        transform_train = transforms.Compose([transforms.Resize(size=image_size), transforms.RandomCrop(size=crop_size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(), transforms.Normalize(mean=[0.4425, 0.4695, 0.3266],
                                                                                          std=[0.2353, 0.2219,
                                                                                               0.2325])])
        transform_test = transforms.Compose([transforms.Resize(size=image_size), transforms.CenterCrop(size=crop_size),
                                             transforms.ToTensor(), transforms.Normalize(mean=[0.4425, 0.4695, 0.3266],
                                                                                         std=[0.2353, 0.2219, 0.2325])])

    trainset = Plantnet(root, 'train', transform=transform_train)
    train_class_to_num_instances = Counter(trainset.targets)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    valset = Plantnet(root, 'val', transform=transform_test)

    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)

    testset = Plantnet(root, 'test', transform=transform_test)
    test_class_to_num_instances = Counter(testset.targets)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    val_class_to_num_instances = Counter(valset.targets)
    n_classes = len(trainset.classes)

    dataset_attributes = {'n_train': len(trainset), 'n_val': len(valset), 'n_test': len(testset),
                          'n_classes': n_classes,
                          'class2num_instances': {'train': train_class_to_num_instances,
                                                  'val': val_class_to_num_instances,
                                                  'test': test_class_to_num_instances},
                          'class_to_idx': trainset.class_to_idx}

    return trainloader, valloader, testloader, dataset_attributes
