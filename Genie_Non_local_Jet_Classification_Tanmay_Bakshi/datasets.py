import os
import requests
import functools
import pathlib
import shutil
import logging
import awkward as ak
import pandas as pd
import numpy as np
import torch
# import tqdm.auto as tqdm
import dask as da
import h5py as hp
from preprocess_dask import _transform, _extract_coords
import pyarrow.parquet as pq
from pathlib import Path
import uproot
import dask.dataframe as dd
import dask_awkward
import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
    Downloading the Dataset
'''

def download(url, filename):
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get('Content-Length', 0))

    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    r.raw.read = functools.partial(r.raw.read, decode_content=True)
    with tqdm.tqdm.wrapattr(r.raw, "read", total=file_size) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)
    return path

test_link = "https://zenodo.org/records/2603256/files/test.h5?download=1"
train_link = "https://zenodo.org/records/2603256/files/train.h5?download=1"
val_link = "https://zenodo.org/records/2603256/files/val.h5?download=1"


def convert(source, destdir, basename, start = None, stop = None, step = None, limit = None):
    """
    Converts the DataFrame into an Awkward array and performs the read-write
    operations for the same. Also performs Batching of the file into smaller
    Awkward files.

    :param source: str, The location to the H5 file with the dataframe
    :param destdir: str, The location we need to write to
    :param basename: str, Prefix for all the output file names
    :param step: int, Number of rows per awkward file, None for all rows in 1 file
    :param limit: int, Number of rows to read.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    df = pd.read_hdf(source, key='table', start = start, stop = stop)
    logging.info('Total events: %s' % str(df.shape[0]))
    if limit is not None:
        df = df.iloc[0:limit]
        logging.info('Restricting to the first %s events:' % str(df.shape[0]))
    if step is None:
        step = df.shape[0]

    idx = 0
    # Generate files as batches based on step size, only 1 batch is default
    iter = 0
    for start in range(0, df.shape[0], step):
        iter = iter+1
        # if os.path.exists(destdir):
        #     shutil.rmtree(destdir)
        if not os.path.exists(destdir):
            os.makedirs(destdir)
        output = os.path.join(destdir, '%s_%d.parquet'%(basename, idx))
        logging.info(output)
        # if os.path.exists(output):
        #     logging.warning('... file already exists: continue ...')
        #     continue
        v = _extract_coords(df, start=start, stop=start+step)
        # arr = ak.Array(v)
        ak.to_parquet(v, output, compression=None)
        logging.info("Parquet file no. ", start, " created.")
        idx += 1
    del df, output


def parquet_handler(source_loc, dest_loc = None):
    parquet_dir = Path(source_loc)
    directory = source_loc.split('/')[-1]
    if not os.path.exists(dest_loc):
        os.makedirs(dest_loc)

    csv_path = str(os.path.join(dest_loc, '%s_processed.csv' % directory))
    if os.path.exists(csv_path):
        logging.info("... CSV file already exists, moving on...")
        pass
    else:
        for i, parquet_path in enumerate(parquet_dir.glob('%s_file_*.parquet' % directory)):
            df = pq.read_table(parquet_path).to_pandas()
            write_header = i == 0 # Write header only on the 0th file
            write_mode = 'w' if i == 0 else 'a' # 'write' mode for 0th file, 'append' for others
            df.to_csv(csv_path, mode=write_mode, header=write_header)

    return

if __name__ == "__main__":

    CURRENT_DIR = os.getcwd()
    print(CURRENT_DIR)
    # Define the new folder name and its path
    new_folder_name = "downloads"
    new_folder_path = os.path.join(CURRENT_DIR, new_folder_name)

    # Check if the folder exists, if not, create it
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path, exist_ok=True)

    # Set the PROJECT_DIR to the new folder path
    PROJECT_DIR = new_folder_path
    PARQUET_FILE_LOC = os.path.join(PROJECT_DIR, 'converted')
    download(test_link, os.path.join(PROJECT_DIR, 'test.h5'))
    download(train_link, os.path.join(PROJECT_DIR, 'train.h5'))
    download(val_link, os.path.join(PROJECT_DIR, 'val.h5'))

    # Call the function
    convert(source = os.path.join(PROJECT_DIR, 'train.h5'), destdir = os.path.join(PROJECT_DIR, 'converted', 'train'), basename = 'train_file', start = 0, stop = 100000, step = 1000, limit = None)
    convert(source = os.path.join(PROJECT_DIR, 'test.h5'), destdir = os.path.join(PROJECT_DIR, 'converted', 'test'), basename = 'test_file', start = 0, stop = 30000, step = 100, limit = None)
    convert(source = os.path.join(PROJECT_DIR, 'val.h5'), destdir = os.path.join(PROJECT_DIR, 'converted', 'val'), basename = 'val_file', start = 0, stop = 10000, step = 1000, limit = None)

""" DO NOT UNCOMMENT THE FOLLOWING SECTION TO RUN THE CODE FOR DEMONSTRATION. THIS SECTION IS FOR FUTURE DEVELOPMENT AND OPTIMIZATION PURPOSES 
    AND IS NOT USEFUL FOR THE CURRENT SCOPE OF THE OBJECTIVE."""
    # print(df['x'][1])
    # parquet_handler(source_loc = os.path.join(PROJECT_DIR, 'converted', 'train'), dest_loc = os.path.join(PROJECT_DIR, 'processed', 'train'))
    # parquet_handler(source_loc = os.path.join(PROJECT_DIR, 'converted', 'test'), dest_loc = os.path.join(PROJECT_DIR, 'processed', 'test'))
    # parquet_handler(source_loc = os.path.join(PROJECT_DIR, 'converted', 'val'), dest_loc = os.path.join(PROJECT_DIR, 'processed', 'val'))

    # root_handler(os.path.join(PROJECT_DIR, 'prep'), os.path.join(PROJECT_DIR, 'prep'))