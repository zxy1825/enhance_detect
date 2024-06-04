'''
Author                        : Zhang Xiuyu
Date                          : 2024-05-14 14:19:20
LastEditTime                  : 2024-05-14 17:30:20
LastEditors                   : Zhang Xiuyu
Description                   : load labels and images
'''

import cv2
import math
import numpy as np
import os
import random
import torch
from torch.utils.data import DataLoader, Dataset, dataloader, distributed

IMG_FORMATS = "jpeg", "jpg", "png"
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders

def create_dataloader(
    path,
    imgsz,
    batch_size,
    shuffle=True,
    augment=False,
    seed=0,
):
    dataset = LoadImagesAndLabels(
            path,
            imgsz,
            batch_size,
            shuffle=shuffle,
            augment=augment, # augmentation
            seed=seed
        )
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size])
    sampler = SmartDistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed)
    return InfiniteDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=LoadImagesAndLabels.collate_fn4,
        worker_init_fn=seed_worker,
        generator=generator,
    ), dataset


def seed_worker(worker_id):
    """
    Sets the seed for a dataloader worker to ensure reproducibility, based on PyTorch's randomness notes.

    See https://pytorch.org/docs/stable/notes/randomness.html#dataloader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SmartDistributedSampler(distributed.DistributedSampler):
    # 自动计算分布式数据加载每个设备对应的索引
    def __iter__(self):
        # 随机数据清洗，根据种子生成随机数
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # determine the the eventual size (n) of self.indices (DDP indices)
        n = int((len(self.dataset) - self.rank - 1) / self.num_replicas) + 1  # num_replicas == WORLD_SIZE
        idx = torch.randperm(n, generator=g)
        if not self.shuffle:
            idx = idx.sort()[0]

        idx = idx.tolist()
        if self.drop_last:
            idx = idx[: self.num_samples]
        else:
            padding_size = self.num_samples - len(idx)
            if padding_size <= len(idx):
                idx += idx[:padding_size]
            else:
                idx += (idx * math.ceil(padding_size / len(idx)))[:padding_size]

        return iter(idx)


class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        """Initializes an InfiniteDataLoader that reuses workers with standard DataLoader syntax, augmenting with a
        repeating sampler.
        """
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler in the InfiniteDataLoader."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Yields batches of data indefinitely in a loop by resetting the sampler when exhausted."""
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        """Initializes a perpetual sampler wrapping a provided `Sampler` instance for endless data iteration."""
        self.sampler = sampler

    def __iter__(self):
        """Returns an infinite iterator over the dataset by repeatedly yielding from the given sampler."""
        while True:
            yield from iter(self.sampler)
            

class LoadImagesAndLabels(Dataset):
    def __init__(self, image_path, batch_size, img_size=640, shuffle=True,
              augment=False, seed=0):
        self.batch_size = batch_size
        self.image_size = img_size
        self.image_path = image_path

    def __len__(self):
        1
    
    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def load_image(self, i):
        """
        Loads an image by index, returning the image, its original dimensions, and resized dimensions.

        Returns (im, original hw, resized hw)
        """
        im, f, fn = (
            self.ims[i],
            self.im_files[i],
            self.npy_files[i],
        )
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                assert im is not None, f"Image Not Found {f}"
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def cache_images_to_disk(self, i):
        """Saves an image to disk as an *.npy file for quicker loading, identified by index `i`."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]))