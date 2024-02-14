import jax
import jax.numpy as jnp
from utils.dataloaders import DecoderImageDataset, JaxBatchDataloader
from datasets import Dataset as HFDataset
import tqdm

_, _, dataset_rng, valset_rng = jax.random.split(jax.random.PRNGKey(0), 4) # just to replicate existing rng

BATCH_SIZE = 16
DATA_ROOT = "/"
TRAIN_CACHE_SIZE = 500000 # Number of batches. ~3 days of training
OUTPUT_DIR = "/path/to/output/dir"

# For these, the CSVs should be nothing other than a line separated list of file paths
hfds_train = HFDataset.from_csv("../sd_vae_trainer/dataset.csv")
# Point to validation set separately, normally we were splitting off the head from the train dataset.
hfds_val = HFDataset.from_csv("../sd_vae_trainer/dataset.csv")

train_ds = DecoderImageDataset(hfds_train, root=DATA_ROOT) # type: ignore
test_ds = DecoderImageDataset(hfds_val, root=DATA_ROOT) # type: ignore

train_dl = JaxBatchDataloader(dataset_rng, BATCH_SIZE, train_ds)
test_dl = JaxBatchDataloader(valset_rng, BATCH_SIZE, test_ds, only_once=True)

for idx, batch in tqdm(enumerate(train_dl), total=TRAIN_CACHE_SIZE, desc="Building image cache (training split)...", dynamic_ncols=True):
    jnp.savez(
        f"{OUTPUT_DIR}/train/batch_{idx}.npz",
        original=batch["original"],
        name=batch["name"]
    )
    if idx == TRAIN_CACHE_SIZE - 1:
        break

for idx, batch in tqdm(enumerate(test_dl), total=len(test_dl), desc="Building image cache (validation split)...", dynamic_ncols=True):
    jnp.savez(
        f"{OUTPUT_DIR}/val/batch_{idx}.npz",
        original=batch["original"],
        name=batch["name"]
    )
    if idx == len(test_dl) - 1:
        break