import math
import random
from io import BytesIO
from typing import List, Tuple
import torch
from pathlib import Path
import os
import numpy as np
from PIL import Image, ImageCms
Image.MAX_IMAGE_PIXELS = None
import torchvision.transforms as T
from torchvision.transforms import functional as F
import tqdm
import jax.numpy as jnp
from queue import Queue
from threading import Thread
import jax

class RandomResizedSquareCrop(T.RandomResizedCrop):
    def __init__(self, size, rng, scale=(0.08, 1.0), ratio=(1.0, 1.0), interpolation=F.InterpolationMode.LANCZOS):
        super().__init__(size, scale=scale, ratio=ratio, interpolation=interpolation)
        self.rng = rng

    def get_params(self, img: torch.Tensor, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
        _, height, width = F.get_dimensions(img)
        area = height * width
        self.rng, attempt_rng = jax.random.split(self.rng, 2)
        for _ in range(10):
            attempt_rng, area_rng, aspect_rng, swap_rng, h_rng, w_rng = jax.random.split(attempt_rng, 6)
            scale_min, scale_max = scale
            size_x, size_y = self.size
            scale_min = max(size_x * size_y / area, scale_min)
            target_area = jax.random.uniform(area_rng, minval=scale_min, maxval=scale_max) * area
            aspect_ratio = math.exp(jax.random.uniform(aspect_rng, minval=math.log(ratio[0]), maxval=math.log(ratio[1])))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if jax.random.uniform(swap_rng) < 0.5:
                w, h = h, w

            if w <= width and h <= height:
                i = jax.random.randint(h_rng, (), 0, height - h)
                j = jax.random.randint(w_rng, (), 0, width - w)
                return i, j, h, w

        # Fallback to central crop if the max attempts are reached
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = int(round(height * min(ratio)))
            h = height
        elif in_ratio > max(ratio):
            h = int(round(width / max(ratio)))
            w = width
        else:
            w = width
            h = height

        i = (height - h) // 2
        j = (width - w) // 2

        return i, j, h, w

class RandomDownscale(torch.nn.Module):
    def __init__(self, rng, sizes=[640, 720, 960, 1080], side="longest", interpolation=F.InterpolationMode.LANCZOS):
        super().__init__()

        self.sizes = sizes
        self.side = side
        self.interpolation = interpolation
        self.rng = rng

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        _, height, width = F.get_dimensions(img)

        match self.side:
            case "width":
                scale_dim = width
            case "height":
                scale_dim = height
            case "shortest":
                scale_dim = min(width, height)
            case "longest":
                scale_dim = max(width, height)

        sizes = [size for size in self.sizes if size >= scale_dim]
        if len(sizes) == 0:
            return img
        self.rng, choice_rng = jax.random.split(self.rng, 2)
        scale = scale_dim / jax.random.choice(choice_rng, sizes)
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))

        return F.resize(img, [new_height, new_width], interpolation=self.interpolation)

class RandomRotate90(T.RandomRotation):
    def __init__(self, rng):
        super().__init__(0, interpolation=F.InterpolationMode.NEAREST, expand=False, center=None, fill=0)
        self.rng = rng

    def __call__(self, img: torch.Tensor):
        self.rng, choice_rng = jax.random.split(self.rng, 2)
        angle = jax.random.choice(choice_rng, [0, 90, 180, 270])
        return F.rotate(img, angle)

class RandomHorizontalFlip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, rng, p=0.5):
        super().__init__()
        self.p = p
        self.rng = rng

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        self.rng, flip_rng = jax.random.split(self.rng, 2)
        if jax.random.uniform(flip_rng) < self.p:
            return F.hflip(img)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomVerticalFlip(torch.nn.Module):
    """Vertically flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, rng, p=0.5):
        super().__init__()
        self.p = p
        self.rng = rng

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        self.rng, flip_rng = jax.random.split(self.rng, 2)
        if jax.random.uniform(flip_rng) < self.p:
            return F.vflip(img)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

class DecoderImageDataset(torch.utils.data.Dataset):
    def __init__(self, hfds, rng, root=None):
        """hdfs: HFDataset"""
        self.rng, crop_rng, rotate_rng, hflip_rng, vflip_rng = jax.random.split(rng, 5)
        self.hfds = hfds
        self.root = root
        self.transform = T.Compose([
            # Random cropping of a square region with size at least 256x256
            RandomResizedSquareCrop(
                size=(256, 256),
                scale=(0.08, 1.0),
                ratio=(1.0, 1.0),
                interpolation=F.InterpolationMode.LANCZOS,
                rng=crop_rng
                ),
            # Random rotation in 90 degree increments
            RandomRotate90(rng=rotate_rng),
            # Random horizontal and vertical flips
            RandomHorizontalFlip(rng=hflip_rng),
            RandomVerticalFlip(rng=vflip_rng),
            # Convert to tensor
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.hfds)

    def __getitem__(self, idx):
        attempt = 0
        self.rng, attempt_rng = jax.random.split(self.rng, 2)
        while True:
            try:
                example = self.hfds[idx]
                # indices = example["indices"]
                path = example["path"]
                if self.root is not None:
                    path = os.path.join(self.root, path.lstrip("/"))
                orig_arr = self.load(path)
                return {
                    # "indices": indices,
                    "original": orig_arr,
                    "name": Path(path).name,
                }
            except:
                attempt_rng, retry_rng = jax.random.split(attempt_rng, 2)
                idx = jax.random.randint(retry_rng, (), 0, len(self.hfds) - 1)
                attempt += 1
                if attempt > 10:
                    print(f"Error reading image at index {idx}: {attempt} attempts done, either you're VERY unlucky or the dataloader broke")

    def load(self, path):
        img = Image.open(path)

        # ensure image is in sRGB color space
        icc_raw = img.info.get('icc_profile')
        if icc_raw:
            img = ImageCms.profileToProfile(
                img,
                ImageCms.ImageCmsProfile(BytesIO(icc_raw)),
                ImageCms.createProfile(colorSpace='sRGB'),
                outputMode='RGB'
            )
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # For downscaling to a fixed set of resolutions, followed by cropping:
        # RandomDownscale(),
        # RandomCrop(size=(256, 256),

        # Apply transformations
        img = self.transform(img)
        img = torch.unsqueeze(img, 0)
        return img.permute(0, 2, 3, 1).numpy()

    @staticmethod
    def collate_fn(examples, return_names=False):
        res = {
            # "indices": [example["indices"] for example in examples],
            "original": np.concatenate(
                [example["original"] for example in examples], axis=0
            ),
        }
        if return_names:
            res["name"] = [example["name"] for example in examples] # type: ignore
        return res

class LatentCacheDataset:
    def __init__(self, data_dir, cache_size, mmap_preload: bool = True):
        self.mmap_preload = mmap_preload
        self.file_list = []
        self.queue = Queue(64)
        for i in tqdm(range(cache_size), desc="Getting file list...", dynamic_ncols=True):
            self.file_list.append(os.path.join(data_dir, f"batch_{i}.npz"))
        self.buffer_thread = Thread(target=self.buffer_thread_function)
        self.buffer_thread.daemon = True  # Daemonize the thread to exit with the main program
        self.buffer_thread.start()
        if mmap_preload:
            self.file_mmap = []
            for filename in tqdm(self.file_list, desc="Building memory map...", dynamic_ncols=True):
                data = jnp.load(filename, mmap_mode='r')
                self.file_mmap.append(data)

    def buffer_thread_function(self):
        while True:
            for filename in self.file_list:
                file = jnp.load(filename)
                dict = {
                    'original': jnp.clip(file['original'].astype(jnp.float32) / 255, 0, 1),
                    'latent_dist': file['latent_dist']
                }
                self.queue.put(dict)

    def __len__(self):
        return len(self.file_list)

    # def __getitem__(self, index):
    #     if self.mmap_preload:
    #         data = self.file_mmap[index]
    #     else:
    #         data = jnp.load(self.file_list[index])
    #     return {'original': data['original'], 'latent': data['latent']}

    def __iter__(self):
        while True:
            yield self.queue.get()
        # for i in range(len(self)):
        #     yield self[i]