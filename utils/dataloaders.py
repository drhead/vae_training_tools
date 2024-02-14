from io import BytesIO
import threading
from pathlib import Path
import os
import numpy as np
import cv2
from PIL import Image, ImageCms
Image.MAX_IMAGE_PIXELS = None
import tqdm
import jax.numpy as jnp
from queue import Queue
from threading import Thread
import jax
from concurrent import futures

class RNGStateWrapper:
    def __init__(self, rng):
        self.rng = rng

class DecoderImageDataset():
    def __init__(self, hfds, root=None):
        """hdfs: HFDataset"""
        self.hfds = hfds
        self.root = root

    def __len__(self):
        return len(self.hfds)

    def get_at_idx(self, idx, rng):
        attempt = 0
        attempt_rng, transforms_rng = jax.random.split(rng.rng)
        while True:
            try:
                example = self.hfds[int(idx)]
                # indices = example["indices"]
                path = example["path"] # type: str
                if self.root is not None:
                    path = os.path.join(self.root, path.lstrip("/"))
                orig_arr = self.load(path, transforms_rng)
                return {
                    # "indices": indices,
                    "original": orig_arr,
                    "name": Path(path).name,
                }
            except:
                attempt_rng, retry_rng = jax.random.split(attempt_rng)
                idx = int(jax.random.randint(retry_rng, (), 0, len(self.hfds) - 1))
                attempt += 1
                if attempt > 10:
                    print(f"Error reading image at index {idx}: {attempt} attempts done, either you're VERY unlucky or the dataloader broke")
                    raise

    def load(self, path: str, rng: jax.Array):

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

        with jax.default_device(jax.devices("cpu")[0]):
            size_rng, crop_rng, fliph_rng, flipw_rng, rot_rng = jax.random.split(rng, 5)
            img_array = np.asarray(img, dtype=np.uint8)

            H, W, C = img_array.shape

            # Random square crop + resize to 256x256
            crop_size = jax.random.randint(size_rng, (), 256, np.minimum(H, W)).item()
            croph_rng, cropw_rng = jax.random.split(crop_rng)
            croph = jax.random.randint(croph_rng, (), 0, H - crop_size).item()
            cropw = jax.random.randint(cropw_rng, (), 0, W - crop_size).item()
            img_array = img_array[croph:croph+crop_size, cropw:cropw+crop_size]
            # img_array = cv2.resize(img_array, (256, 256), interpolation=cv2.INTER_AREA)
            img_array = np.asarray(Image.fromarray(img_array).resize((256, 256), Image.Resampling.LANCZOS, reducing_gap=3.0))

            # Random flips, both axes
            if jax.random.uniform(fliph_rng) > 0.5:
                img_array = np.flip(img_array, 0)
            if jax.random.uniform(flipw_rng) > 0.5:
                img_array = np.flip(img_array, 2)

            # Random rotation
            rotate_times = jax.random.randint(rot_rng, (), 0, 3).item()
            img_array = np.rot90(img_array, rotate_times)
            img_array = np.expand_dims(img_array, 0)
            img_array = jnp.asarray(img_array, dtype=jnp.bfloat16) / 255 # type: jax.Array

        return img_array

    @staticmethod
    def collate_fn(examples, return_names=False):
        res = {
            # "indices": [example["indices"] for example in examples],
            "original": jnp.concatenate(
                [example["original"] for example in examples], axis=0
            ),
        }
        if return_names:
            res["name"] = [example["name"] for example in examples] # type: ignore

        return res

class JaxBatchSampler:
    def __init__(self, rng, batch_size, ds_len, only_once):
        self.rng = rng
        self.batch_size = batch_size
        self.dataset_len = ds_len
        self.only_once = only_once

    def __iter__(self):
        with jax.default_device(jax.devices("cpu")[0]):
            while True:
                self.rng, shuffle_rng = jax.random.split(self.rng)
                shuffled_idx = jax.random.permutation(shuffle_rng, jnp.arange(0, self.dataset_len), independent=True)
                shuffled_idx = np.asarray(shuffled_idx)

                while len(shuffled_idx) >= self.batch_size:

                    yield shuffled_idx[:self.batch_size]
                    shuffled_idx = shuffled_idx[self.batch_size:]
                if self.only_once:
                    break


class JaxBatchDataloader:
    def __init__(self, rng, batch_size, dataset, only_once = False):
        self.rng, sampler_rng = jax.random.split(rng)
        self.batch_size = batch_size
        self.dataset = dataset # type: DecoderImageDataset
        self.sampler = JaxBatchSampler(sampler_rng, batch_size, len(dataset), only_once)
        self.batchqueue = Queue(64)
        self.batch_thread = threading.Thread(target=self.loader_thread)
        self.batch_thread.start()

    def loader_thread(self):
        with futures.ThreadPoolExecutor(max_workers=64) as executor:
            for batch in self.sampler:
                self.rng, batch_rng = jax.random.split(self.rng)
                future = executor.submit(self.prep_batch, batch, batch_rng)
                self.batchqueue.put(future)

    def prep_batch(self, batch, rng):
        with futures.ThreadPoolExecutor(max_workers=64) as executor:
            batch_rng_arr = jax.random.split(rng, self.batch_size)
            rng_states = [RNGStateWrapper(rng) for rng in batch_rng_arr]
            batch_data = list(executor.map(self.dataset.get_at_idx, batch, rng_states))
            return self.dataset.collate_fn(batch_data, return_names=True)

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        while True:
            # print(f"queue size {self.batchqueue.qsize()}")
            yield self.batchqueue.get().result()

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