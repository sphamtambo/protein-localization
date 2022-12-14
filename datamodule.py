import numpy as np
import pandas as pd
import torch
import transformers
import os
import requests
import re

from config import *
from tqdm.auto import tqdm

class DeepLocDataset():
    """ 
    Download and loads the dataset from csv files passed to the parser
    :param train: flag to return the train set.
    :param valid: flag to return the valid set.
    :param test: flag to return the test set.
    Returns:
        -Training, Validation, and Testing Dataset
    """

    def __init__(self):
        self.DownloadDeeplocDataset()

    def download_file(self, url, filename):
        """ retrieve data from the specifies url. """
        response = requests.get(url, stream=True)
        with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                          total=int(response.headers.get("content-length", 0)),
                          desc=filename) as fout:
            for chunk in response.iter_content(chunk_size=4096):
                fout.write(chunk)


    def DownloadDeeplocDataset(self):
        """ download dataset from the provided url(s). """
        deeplocDatasetTrainUrl = 'https://rostlab.org/~deepppi/deeploc_data/deeploc_our_train_set.csv'
        deeplocDatasetValidUrl = 'https://rostlab.org/~deepppi/deeploc_data/deeploc_our_val_set.csv'
        deeplocDatasetTestUrl = 'https://rostlab.org/~deepppi/deeploc_data/deeploc_test_set.csv'


        datasetFolderPath = 'data/'
        self.trainFilePath = os.path.join(datasetFolderPath, "deeploc_per_protein_train.csv")
        self.validFilePath = os.path.join(datasetFolderPath, "deeploc_per_protein_valid.csv")
        self.testFilePath = os.path.join(datasetFolderPath, "deeploc_per_protein_test.csv")


        if not os.path.exists(datasetFolderPath):
            os.makedirs(datasetFolderPath)


        if not os.path.exists(self.trainFilePath):
            self.download_file(deeplocDatasetTrainUrl, self.trainFilePath)
        if not os.path.exists(self.validFilePath):
            self.download_file(deeplocDatasetValidUrl, self.validFilePath)
        if not os.path.exists(self.testFilePath):
            self.download_file(deeplocDatasetTestUrl, self.testFilePath)

    def load_df(self, df_name = "train"):
        """ load dataset """

        if df_name == "train":
            path = self.trainFilePath
        elif df_name == "valid":
            path = self.validFilePath
        else:
            path = self.testFilePath

        df = pd.read_csv(path,
                        usecols = [0,1],
                        names=["sequence", "location"],
                       skiprows=0)

        categories = df["location"].astype("category").cat
        df["categories"] = categories.codes
        df = df.drop('location', axis=1)
        print(f"{df_name} downloaded")
        return df


class ProteinSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, sequence, labels, tokenizer, max_length):
        self.sequence = sequence
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        sequence = str(self.sequence[idx])
        ### make sure there is a space between avery token 
        sequence = " ".join(sequence)
        ### map rarely occuring amino acids to X
        sequence = re.sub(r"[UZOB]", "x", sequence)

        labels = int(self.labels[idx])

        encodings = self.tokenizer.encode_plus(
            sequence,
            truncation=True,
            add_special_tokens=True,
            padding="max_length", 
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt"
            )


        return {
                "input_ids": encodings["input_ids"].flatten(),
                "attention_mask": encodings["attention_mask"].flatten(),
                "labels": torch.tensor(labels, dtype=torch.long)
                }

dataset = DeepLocDataset()
train = dataset.load_df("train")
valid = dataset.load_df("valid")
test = dataset.load_df("test")

train_dataset = ProteinSequenceDataset(
        sequence=train.sequence.to_numpy(),
        labels=train.categories.to_numpy(),
        tokenizer=TOKENIZER,
        max_length=MAX_LEN)

valid_dataset = ProteinSequenceDataset(
        sequence=valid.sequence.to_numpy(),
        labels=valid.categories.to_numpy(),
        tokenizer=TOKENIZER,
        max_length=MAX_LEN)

test_dataset = ProteinSequenceDataset(
        sequence=test.sequence.to_numpy(),
        labels=test.categories.to_numpy(),
        tokenizer=TOKENIZER,
        max_length=MAX_LEN)


train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS)

valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=VALID_BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS)

test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=TEST_BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS)



