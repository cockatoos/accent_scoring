#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import os
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence
import torch
import torch.nn as nn
import pickle
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path



# In[51]:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--orig_data_path',
        type=str,
        help='Path to the original recordings before fragmentating',
        default='/Users/yejinseo/Desktop/azure_ai_hack/accent_scoring/data/recordings/recordings'
    )
    parser.add_argument(
        '--train_data_path',
        type=str,
        help='Path to the training data',
        default='/Users/yejinseo/Desktop/azure_ai_hack/accent_scoring/data/train'
    )

    parser.add_argument(
        '--val_data_path',
        type=str,
        help='Path to the validation data of .wav',
        default='/Users/yejinseo/Desktop/azure_ai_hack/accent_scoring/data/val'
    )

    parser.add_argument(
        '--test_data_path',
        type=str,
        help='Path to the testing data of .wav',
        default='/Users/yejinseo/Desktop/azure_ai_hack/accent_scoring/data/test'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        help='Path to the model to be saved',
        default='/Users/yejinseo/Desktop/azure_ai_hack/accent_scoring/model'
    )

    args = parser.parse_args()


# In[48]:


class MFCCDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.mfccs = []
        self.labels = []

        file_list = [x for x in os.listdir(self.root_dir) if x.startswith('data_batch')]
        for i in range(len(file_list)):

            f = file_list[i]
            with open(self.root_dir / f, 'rb') as handle:
                entry = pickle.load(handle)
            self.mfccs.append(entry['data'])
            self.labels.extend(entry['labels'])
            #data = torch.load(self.root_dir / f)

        self.mfccs = torch.from_numpy(np.vstack(self.mfccs).reshape(-1,3,20,22)).float()
        self.labels = torch.tensor(self.labels, dtype=torch.float)

    def __len__(self):

        return len(self.mfccs)

    def __getitem__(self, idx):

        return self.mfccs[idx], self.labels[idx]


# In[49]:

original_dir = Path(args.orig_data_path)
train_dir =  Path(args.train_data_path)
val_dir =  Path(args.val_data_path)
test_dir =  Path(args.test_data_path)
print("===== DATA =====")
print("recordings DATA PATH: " + args.orig_data_path)
print("train DATA PATH: " + args.train_data_path)
print("val DATA PATH: " + args.val_data_path)
print("test DATA PATH: " + args.test_data_path)

print("================")
files = os.listdir(original_dir)


other_accent_types = ["mandarin", "japanese", "korean", "taiwanese", "cantonese", "thai", "indonesian"]

english_accent_files = []
other_accent_files = []

num_train_files = 150
num_val_files = 23
num_test_files = 25

end_idx_train = num_train_files
end_idx_val = end_idx_train + num_val_files
end_idx_test = end_idx_val + num_test_files

for f in files:
    if "english" in f:
        english_accent_files.append(f)

    if any(t in f for t in other_accent_types):
        other_accent_files.append(f)

np.random.shuffle(english_accent_files)
np.random.shuffle(other_accent_files)

print(f"Number of english accent files: {len(english_accent_files)}")
print(f"Number of other accent files: {len(other_accent_files)}")

train_files = english_accent_files[0:end_idx_train] + other_accent_files[0:end_idx_train]
val_files   = english_accent_files[end_idx_train:end_idx_val] + other_accent_files[end_idx_train:end_idx_val]
test_files  = english_accent_files[end_idx_val:end_idx_test] + other_accent_files[end_idx_val:end_idx_test]

np.random.shuffle(train_files)
np.random.shuffle(val_files)
np.random.shuffle(test_files)

print(f"Number of training files: {len(train_files)}")
print(f"Number of validaiton files: {len(val_files)}")
print(f"Number of test files: {len(test_files)}")


# In[4]:


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


# In[50]:


def generate_model_data(path, files, train):

    counter = 0
    seg_thresh = 500
    batch_num = 1
    items = []
    labels = []

    for f in files:

        if "english" in f:
            label = 1
        else:
            label = 0

        sound_file = AudioSegment.from_mp3(original_dir / f)
        audio_chunks = split_on_silence(sound_file,
            # must be silent for at least half a second
            min_silence_len = 80,

            # consider it silent if quieter than -16 dBFS
            silence_thresh=-30
        )


        for seg in audio_chunks:

            seg_len = len(seg)

            if seg_len >= seg_thresh:
                seg_standardized = seg[0:seg_thresh]
            else:
                seg_standardized = seg + AudioSegment.silent(duration=(seg_thresh - seg_len))


            samples = seg_standardized.get_array_of_samples()
            arr = np.array(samples).astype(np.float32)/32768 # 16 bit
            arr = librosa.core.resample(arr, seg_standardized.frame_rate, 22050, res_type='kaiser_best')

            mfcc = librosa.feature.mfcc(y=arr, sr=22050)
            data = generate_mfcc_data(mfcc)
            #torch.save(torch.from_numpy(data).float(), path / f"{label}_word{counter}.pt")
            items.append(data)
            labels.append(label)
            #counter += 1

            if train:
                noise = np.random.normal(0,1, mfcc.shape)
                mfcc_noisy = mfcc + noise
                noisy_data = generate_mfcc_data(mfcc_noisy)
                items.append(noisy_data)
                labels.append(label)
                #torch.save(torch.from_numpy(noisy_data).float(), path / f"{label}_word{counter}.pt")
                #counter += 1

    max_batch_size = len(labels) // 5
    for j in range(0,len(items),max_batch_size):
        curr_data = items[j:j + max_batch_size]
        curr_labels = labels[j:j + max_batch_size]
        batch_mfcc = np.vstack(curr_data).reshape(-1,3,20,22)
        entry = dict()
        entry['data'] = batch_mfcc
        entry['labels'] = curr_labels
        with open(path / f'data_batch_{batch_num}.pickle', 'wb') as handle:
            pickle.dump(entry, handle, protocol=pickle.HIGHEST_PROTOCOL)
        batch_num += 1





#Uncomment to create the files for the dataset folders

# generate_model_data(val_dir, val_files, False)
# print("Validaiton words created")
# generate_model_data(test_dir, test_files, False)
# print("Testing words created")
# generate_model_data(train_dir, train_files, True)
# print("Training words created")


train_data = MFCCDataset(train_dir)
val_data = MFCCDataset(val_dir)
# test_data = MFCCDataset(test_data_dir)


# In[60]:


print(len(train_data.mfccs))


# In[133]:


print(len(train_data))


# In[55]:


print(len(train_data))


# In[57]:


print(train_data[0][0].shape)


# In[54]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[55]:


#Model definition
model = nn.Sequential(
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
            nn.Linear(768,128),
            nn.Dropout(0.5),
            nn.Linear(128,1),
            nn.Sigmoid()
            # nn.Softmax(dim=0)
        ).to(device)


# In[26]:


print(model)


# In[56]:


with torch.no_grad():
    torch.cuda.empty_cache()


# In[61]:


train_loader = DataLoader(train_data,batch_size=32,shuffle=True)
val_loader = DataLoader(val_data,batch_size=32,shuffle=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 100
max_val_acc = 0

for epoch in range(epochs):

    running_loss = 0
    correct = 0
    for i, (inputs, labels) in enumerate(train_loader):
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

    if epoch % 10 == 0:
        with torch.no_grad():
            val_loss = 0
            val_correct = 0
            for j, (d,l) in enumerate(val_loader):
                o = model(d.to(device))
                loss = nn.BCELoss()(o,l.to(device).reshape(-1,1))
                val_loss += loss.item()
                o = o.reshape(1,-1)
                o = o.squeeze()
                for i in range(o.size()[0]):
                    if (l[i] == 0 and o[i] < 0.5) or (l[i] == 1 and o[i] >= 0.5):
                        val_correct += 1

            accuracy = 100 * val_correct / len(val_data)
            if accuracy > max_val_acc:
                print('new record making model!')
                max_val_acc = accuracy
                torch.save(model.state_dict(), args.model_path+'/model.pt')
            print(f"Validation loss for epoch {epoch}: {val_loss / len(val_loader)}, accuracy: {accuracy}")

    accuracy = 100 * correct / len(train_data)
    print(f"Epoch:{epoch}, avg loss for epoch:{running_loss / len(train_loader)}, accuracy: {accuracy}")
