import torchaudio
from torch.utils.data import Dataset

import pandas as pd
import os

class UrbanSoundDataset(Dataset):

    def __init__(self, annotations_file, audio_dir):
        self.annotations = pd.read_csv(annotations_file)
        self.audi_dir = audio_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path, format="wav")

        return signal, label 

    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audi_dir, fold, self.annotations.iloc[index, 0])

        return path
    
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]
    
if __name__ == "__main__":
    ANNOTATIONS_FILE = "/Users/milindsoorya/datasets/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "/Users/milindsoorya/datasets/UrbanSound8K/audio"

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR)

    print(f"There are {len(usd)} samples in the dataset")

    signal, label = usd[0]
