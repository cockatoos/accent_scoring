#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import librosa
from pydub import AudioSegment, silence
from pydub.silence import split_on_silence
import torch
import torch.nn as nn
import pickle
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from pathlib import Path
from azureml.core import Dataset
import torch.onnx

import argparse
# import joblib

# In[2]:

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_data_path',
        type=str,
        default='/Users/yejinseo/Desktop/azure_ai_hack/accent_scoring/data/archive/cv-valid-train',
        help='training data path'
    )
    parser.add_argument(
        '--test_data_path',
        type=str,
        default='//Users/yejinseo/Desktop/azure_ai_hack/accent_scoring/data/archive/cv-valid-test',
        help='source testing data path'
    )
    parser.add_argument(
        '--train_csv',
        type=str,
        default='/Users/yejinseo/Desktop/azure_ai_hack/accent_scoring/data/archive/cv-valid-train.csv',
        help='training data csv'
    )
    parser.add_argument(
        '--test_csv',
        type=str,
        default='/Users/yejinseo/Desktop/azure_ai_hack/accent_scoring/data/archive/cv-valid-test.csv',
        help='testing data csv'
    )
    parser.add_argument(
        '--proc_train_str_path',
        type=str,
        default='/Users/yejinseo/Desktop/azure_ai_hack/accent_scoring/data/common_voice/train',
        help='path where pre-precessed training data would be stored'
    )
    parser.add_argument(
        '--proc_test_str_path',
        type=str,
        default='/Users/yejinseo/Desktop/azure_ai_hack/accent_scoring/data/common_voice/test',
        help='path where pre-precessed testing data would be stored'
    )

    args = parser.parse_args()


def generate_dataset(df, files, accents, sizes):
    for accent in accents:
        if accent == 'us':
            label = 1
        else:
            label = 0
        accent_df = df[df['accent'] == accent]
        filenames = accent_df['filename'].tolist()
        if accent in sizes:
            filenames = filenames[:sizes[accent]]
        for name in filenames:
            files.append((name,label))


train_dir =  Path(args.proc_train_str_path)
test_dir =  Path(args.proc_test_str_path)

data_train_src =  Path(args.train_data_path)
data_test_src =  Path(args.test_data_path)

train_files = []
val_files = []
test_files = []

train_df = pd.read_csv(Path(args.train_csv))
test_df = pd.read_csv(Path(args.test_csv))

accents = ['malaysia', 'african', 'wales', 'philippines','hongkong','singapore', 'indian', 'us']
train_sizes = {'us': 6000, 'indian': 4000}
generate_dataset(train_df, train_files, accents, train_sizes)
test_sizes = {'us': 150}
generate_dataset(test_df, test_files, accents, test_sizes)


print(f"Number of training files: {len(train_files)}")
print(f"Number of test files: {len(test_files)}")

def generate_mfcc_data(mfcc):
      mfcc_standardized = np.zeros(mfcc.shape)
      for b in range(mfcc.shape[0]):
          mfcc_slice = mfcc[b,:]
          centered = mfcc_slice - np.mean(mfcc_slice)
          if np.std(centered) != 0:
              centered_scaled = centered / np.std(centered)

          mfcc_standardized[b,:] = centered_scaled

      delta1 = librosa.feature.delta(mfcc_standardized, order=1)
      delta2 = librosa.feature.delta(mfcc_standardized, order=2)
      mfcc_data = np.stack((mfcc_standardized,delta1,delta2))

      return mfcc_data


def segment_and_standardize_audio(path, seg_size):
    sound_file = AudioSegment.from_mp3(path)
    limit = len(sound_file) // seg_size if len(sound_file) % seg_size == 0 else len(sound_file) // seg_size + 1
    chunks = []
    for i in range(0,limit):
        chunk = sound_file[i * seg_size : (i + 1) * seg_size]
        if len(chunk) < seg_size:
            chunk = chunk + AudioSegment.silent(duration=(seg_size - len(chunk)))


        if np.count_nonzero(chunk.get_array_of_samples()) > 45000:
            chunks.append(chunk)
    return chunks



def generate_model_data(src, dst, files, train, mean=0, std=1):

    counter = 0
    seg_size = 1000
    batch_num = 1
    mfccs = []
    items = []
    labels = []
    n_mfcc = 50
    mfcc_width = 44
    c = 0

    for f in files:


        # use for common voice data
        label = f[1]
        audio_chunks = segment_and_standardize_audio(src / f[0], seg_size)
        for seg in audio_chunks:

            samples = seg.get_array_of_samples()
            arr = np.array(samples).astype(np.float32)/32768 # 16 bit
            arr = librosa.core.resample(arr, seg.frame_rate, 22050, res_type='kaiser_best')

            mfcc = librosa.feature.mfcc(y=arr, sr=22050, n_mfcc=n_mfcc)
            mfccs.append(mfcc)
            labels.append(label)

        c += 1
        if c % 100 == 0:
            print(f"Processed {c} files")


    all_data = np.vstack(mfccs).reshape(-1,n_mfcc,mfcc_width)
    if train:
        mean = all_data.mean(axis=0)
        std = all_data.std(axis=0)
        all_data = (all_data - mean) / std
    else:
        all_data = (all_data - mean) / std

    for j in range(all_data.shape[0]):
        d = generate_mfcc_data(all_data[j])
        items.append(d)


    batch_size = len(labels) // 6
    for j in range(6):

        start = j * batch_size
        end = start + batch_size
        if j == 5 and len(labels) % 6 != 0:
            end = len(labels)
        curr_data = items[start:end]
        curr_labels = labels[start:end]
        batch_mfcc = np.vstack(curr_data).reshape(-1,3,n_mfcc,mfcc_width)
        entry = dict()
        entry['data'] = batch_mfcc
        entry['labels'] = curr_labels
        with open(dst / f'data_batch_{j+1}.pickle', 'wb') as handle:
            pickle.dump(entry, handle, protocol=pickle.HIGHEST_PROTOCOL)


    if train:
        return mean, std

mean, std = generate_model_data(data_train_src, train_dir, train_files, True)
print("Training words created")
generate_model_data(data_test_src, test_dir, test_files, False, mean, std)
print("Testing words created")
