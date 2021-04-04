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
        default='/Users/yejinseo/Desktop/azure_ai_hack/accent_scoring/data/common_voice/train',
        help='training data path'
    )
    parser.add_argument(
        '--test_data_path',
        type=str,
        default='/Users/yejinseo/Desktop/azure_ai_hack/accent_scoring/data/common_voice/test',
        help='source testing data path'
    )
    # parser.add_argument(
    #     '--train_csv',
    #     type=str,
    #     default='common_voice_train_csv',
    #     help='training data csv'
    # )
    # parser.add_argument(
    #     '--test_csv',
    #     type=str,
    #     default='common_voice_test_csv',
    #     help='testing data csv'
    # )

    args = parser.parse_args()

class MFCCDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.mfccs = []
        self.labels = []

        for i in range(len(os.listdir(self.root_dir))):

            f = os.listdir(self.root_dir)[i]
            with open(self.root_dir / f, 'rb') as handle:
                entry = pickle.load(handle)
            self.mfccs.append(entry['data'])
            self.labels.extend(entry['labels'])

        self.mfccs = torch.from_numpy(np.vstack(self.mfccs).reshape(-1,3,50,44)).float()
        self.labels = torch.tensor(self.labels, dtype=torch.float)

    def __len__(self):

        return len(self.mfccs)

    def __getitem__(self, idx):

        return self.mfccs[idx], self.labels[idx]


# In[52]:


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


# In[75]:


#Data setup for kaggle common voice dataset

# train_dir =  Path(args.train_data_path)
# test_dir =  Path(args.test_data_path)
#
# data_train_src =  Path('/Users/yejinseo/Desktop/azure_ai_hack/accent_scoring/data/archive/cv-valid-train')
# data_test_src =  Path('/Users/yejinseo/Desktop/azure_ai_hack/accent_scoring/data/archive/cv-valid-test')

# train_csv = Path('/Users/yejinseo/Desktop/azure_ai_hack/accent_scoring/data/archive/cv-valid-train.csv')
# test_csv = Path('/Users/yejinseo/Desktop/azure_ai_hack/accent_scoring/data/archive/cv-valid-test.csv')
#
# train_files = []
# val_files = []
# test_files = []
#
# train_df = pd.read_csv(train_csv)
# test_df = pd.read_csv(test_csv)
#
#
# accents = ['malaysia', 'african', 'wales', 'philippines','hongkong','singapore', 'indian', 'us']
# train_sizes = {'us': 6000, 'indian': 4000}
# generate_dataset(train_df, train_files, accents, train_sizes)
# test_sizes = {'us': 150}
# generate_dataset(test_df, test_files, accents, test_sizes)


# print(f"Number of training files: {len(train_files)}")
# print(f"Number of test files: {len(test_files)}")


# In[63]:


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


# In[64]:


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


# In[65]:


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


# In[76]:


#Uncomment to create the files for the dataset folders (common voice)

# mean, std = generate_model_data(data_train_src, train_dir, train_files, True)
# print("Training words created")
# generate_model_data(data_test_src, test_dir, test_files, False, mean, std)
# print("Testing words created")


# In[77]:


train_data_dir = Path(args.train_data_path)
test_data_dir = Path(args.test_data_path)

train_data = MFCCDataset(train_data_dir)
test_data = MFCCDataset(test_data_dir)


# In[17]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[71]:


class AccentClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3,32,3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            nn.Flatten(1,3),
            nn.Linear(6336,256),
            nn.Dropout(0.5),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


# In[73]:


epochs = 100
kfold = KFold(n_splits=10, shuffle=True)
best_accuracy = 0
for fold, (train_ids, test_ids) in enumerate(kfold.split(train_data)):
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    trainloader = torch.utils.data.DataLoader(
                      train_data,
                      batch_size=128, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(
                      train_data,
                      batch_size=32, sampler=test_subsampler)

    model = AccentClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    for epoch in range(epochs):

        running_loss = 0
        correct = 0
        for i, (inputs, labels) in enumerate(trainloader):
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = nn.BCELoss()(outputs,labels.to(device).reshape(-1,1))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            outputs = outputs.reshape(1, -1)
            outputs = outputs.squeeze()
            for i in range(outputs.size()[0]):
                if (labels[i] == 0 and outputs[i] < 0.5) or (labels[i] == 1 and outputs[i] >= 0.5):
                    correct += 1


        print(f"Epoch {epoch + 1}  Loss: {running_loss / len(trainloader)}  Accuracy: {100 * correct / len(train_ids)}")


    with torch.no_grad():
            model.eval()
            test_loss = 0
            test_correct = 0
            for j, (d,l) in enumerate(testloader):
                o = model(d.to(device))
                loss = nn.BCELoss()(o,l.to(device).reshape(-1,1))
                test_loss += loss.item()
                o = o.reshape(1,-1)
                o = o.squeeze()
                for i in range(o.size()[0]):
                    if (l[i] == 0 and o[i] < 0.5) or (l[i] == 1 and o[i] >= 0.5):
                        test_correct += 1

            accuracy = 100 * test_correct / len(test_ids)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                os.makedirs('outputs', exist_ok=True)
                # note file saved in the outputs folder is automatically uploaded into experiment record
                x = torch.randn(1, 3, 50, 44, requires_grad=True).to(device)
                torch.onnx.export(model, x, "outputs/binary_accent_classifier.onnx", opset_version=11)
                # joblib.dump(value=clf, filename='outputs/sklearn_mnist_model.pkl')

    print(f"Model test accuracy for fold {fold}: {accuracy} ")


# In[174]:


test_loader = DataLoader(test_data,batch_size=32,shuffle=True)
with torch.no_grad():
    model.eval()
    test_loss = 0
    test_correct = 0
    for j, (d,l) in enumerate(test_loader):
        o = model(d.to(device))
        loss = nn.BCELoss()(o,l.to(device).reshape(-1,1))
        val_loss += loss.item()
        o = o.reshape(1,-1)
        o = o.squeeze()
        for i in range(o.size()[0]):
            if (l[i] == 0 and o[i] < 0.5) or (l[i] == 1 and o[i] >= 0.5):
                test_correct += 1

    accuracy = 100 * test_correct / len(test_data)

print(accuracy)


# In[41]:


#Classifying specific set of audio samples
def predict(test_dir):

    predictions = dict()
    for f in os.listdir(test_dir):
        audio_chunks = segment_and_standardize_audio(test_dir / f, 1000)
        num_american_pred = 0
        for seg in audio_chunks:

            samples = seg.get_array_of_samples()
            arr = np.array(samples).astype(np.float32)/32768 # 16 bit
            arr = librosa.core.resample(arr, seg.frame_rate, 22050, res_type='kaiser_best')

            mfcc = librosa.feature.mfcc(y=arr, sr=22050, n_mfcc=50)
            data = generate_mfcc_data(mfcc)
            pred = model(torch.from_numpy(data).unsqueeze(0).float().to(device)).item()
            if pred > 0.5:
                num_american_pred += 1

        frac_american_preds = num_american_pred / len(audio_chunks)

        if frac_american_preds >= 0.5:
            predictions[f] = 1
        else:
            predictions[f] = 0

    return predictions


# In[ ]:


def save_model(model):
    x = torch.randn(1, 3, 50, 44, requires_grad=True).to(device)
    torch.save(model.state_dict(), "binary_accent_classifier.pt")
    torch.onnx.export(model, x, "binary_accent_classifier.onnx", opset_version=11)
