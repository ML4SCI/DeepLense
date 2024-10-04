
import os
import sys
import ast
import requests
import functools
import pathlib
import shutil
import logging
import awkward as ak
import pandas as pd
import numpy as np
from pathlib import Path
import dask.dataframe as dd
from pyarrow import csv

import pickle


def parquet_reader(source_loc):
    parquet_dir = Path(source_loc)
    directory = source_loc.split('/')[-1]
    parq_list = [prq for prq in parquet_dir.glob('%s_file_*.parquet' % directory)]
    ddf = dd.read_parquet(parq_list, engine='pyarrow')
    df = ddf.compute()
    return df


def dataframe_parser(df):
    def process_column(column):
        # Convert each string value in the column to a list of floats
        def clean_and_convert(value):
            # Remove the outer brackets and split by space
            elements = value[1:-1].split(' ')
            # Filter out empty strings, strip '\n', and convert to float
            cleaned_elements = [el.replace('\n', '') for el in elements if el.strip() != '']
            return cleaned_elements

        return column.apply(clean_and_convert)

    # Apply the processing function to each column where the type is 'object'
    processed_df = df.apply(lambda col: process_column(col) if col.dtype == 'object' else col)

    return processed_df

def awkward_structure_parser(df):
    ''' Takes in the pd.DataFrame object read from parquet files, and interprets its columnar
    data as Awkward Arrays. Also, breaks down the chunks into event-wise entries using the `jet_nparticles`
    column to traverse the other fields and splitting them.'''
    awkward_columns = {col: ak.Array(df[col]) for col in df.columns}
    jet_parts = awkward_columns['jet_nparticles']

    event_ctr = 0
    event_dicts = {}
    for i, jet in enumerate(jet_parts):
        start_indices = np.cumsum(np.concatenate(([0], jet[:-1])))
        end_indices = np.cumsum(jet)
        # print(start_indices)
        # print(end_indices)
        # print("OK")
        for event_id, (start, end) in enumerate(zip(start_indices, end_indices)):
            # Extracting all the data for the current event
            event_data = {
                'x' : awkward_columns['x'][i][start:end],
                'y' : awkward_columns['y'][i][start:end],
                'z' : awkward_columns['z'][i][start:end],
                't' : awkward_columns['t'][i][start:end],
                'rho' : awkward_columns['rho'][i][start:end],
                'eta' : awkward_columns['eta'][i][start:end],
                'phi' : awkward_columns['phi'][i][start:end],
                'pt' : awkward_columns["jet_pt"][i][start:end],
                'theta' : awkward_columns['theta'][i][start:end],
                'label' : awkward_columns['label'][i][event_id],
                'train_val_test' : awkward_columns['train_val_test'][i][event_id],
                'n_parts' : jet[event_id]
            }
            event_dicts[f'event_{event_ctr}'] = event_data
            event_ctr += 1

    # input()
    # Create a new DataFrame with Awkward Arrays
    # awkward_df = pd.DataFrame(awkward_columns)

    return event_dicts







if __name__ == '__main__':
    datapath = os.path.join(os.getcwd(), 'downloads/converted/')
    file_cont = os.path.join(datapath, 'train')
    df = parquet_reader(file_cont)
    awkward_dict = awkward_structure_parser(df)

    train_path = os.path.join(os.getcwd(), 'downloads/processed/')
    if os.path.exists(train_path):
        pass
    else:
        os.mkdir(train_path)


    with open(os.path.join(train_path, 'train_data.pkl'), 'wb') as f:
        pickle.dump(awkward_dict, f)
    del df

    file_cont = os.path.join(datapath, 'test')
    df = parquet_reader(file_cont)
    awkward_dict = awkward_structure_parser(df)


    test_path = os.path.join(os.getcwd(), 'downloads/processed/')
    if os.path.exists(train_path):
        pass
    else:
        os.mkdir(train_path)

    with open(os.path.join(test_path, 'test_data.pkl'), 'wb') as f:
        pickle.dump(awkward_dict, f)
    del df

    del train_path, test_path, awkward_dict


    file_cont = os.path.join(datapath, 'val')
    df = parquet_reader(file_cont)
    awkward_dict = awkward_structure_parser(df)

    val_path = os.path.join(os.getcwd(), 'downloads/processed/')
    if os.path.exists(val_path):
        pass
    else:
        os.mkdir(val_path)

    with open(os.path.join(val_path, 'val_data.pkl'), 'wb') as f:
        pickle.dump(awkward_dict, f)
    del df

    del val_path, awkward_dict