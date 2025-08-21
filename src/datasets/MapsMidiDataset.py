import torch.utils
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.functional as F
import torchaudio
import pretty_midi
import pandas as pd

class MapsMidiDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, sample_rate, midi_rate, sample_length, dilation, normalize=True, mode: str = "train", year: str = None):
        if data_dir[-1] != "/":
            data_dir = data_dir + "/"
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.midi_rate = midi_rate
        self.sample_length = sample_length
        self.dilation = dilation
        self.audio_indices = []
        self.midi_indices = []
        self.audio = []
        self.midi = []
        self.year = []
        if normalize:
            self.audio_mean = -1.523106220702175e-05
            self.audio_std = 0.018614452332258224
        else:
            self.audio_mean = 0
            self.audio_std = 1
        
        if year == 'ambient' or year == 'close':
            if year == 'ambient':
                csv_path = f"{self.data_dir}MAPS_ambient.csv"
            elif year == 'close':
                csv_path = f"{self.data_dir}MAPS_close.csv"
            df = pd.read_csv(csv_path)
            train_files = df[df['split'] == 'train']['audio_filename'].str.replace('.wav', '').tolist()
            test_files = df[df['split'] == 'test']['audio_filename'].str.replace('.wav', '').tolist()
            valid_files = df[df['split'] == 'validation']['audio_filename'].str.replace('.wav', '').tolist()
        
        elif year == 'all':
            csv_path = f"{self.data_dir}MAPS_close.csv"
            df_close = pd.read_csv(csv_path)
            csv_path = f"{self.data_dir}MAPS_ambient.csv"
            df_ambient = pd.read_csv(csv_path)
            df = pd.concat([df_close, df_ambient])

            train_files = df[df['split'] == 'train']['audio_filename'].str.replace('.wav', '').tolist()
            test_files = df[df['split'] == 'test']['audio_filename'].str.replace('.wav', '').tolist()
            valid_files = df[df['split'] == 'validation']['audio_filename'].str.replace('.wav', '').tolist()
        elif year == 'single':
            test_files = ["ENSTDkAm/MUS/MAPS_MUS-bk_xmas1_ENSTDkAm"]
            train_files =  ["ENSTDkAm/MUS/MAPS_MUS-bk_xmas1_ENSTDkAm"]
            valid_files =  ["ENSTDkAm/MUS/MAPS_MUS-bk_xmas1_ENSTDkAm"]
        else:
            raise ValueError("Invalid year")

        if mode == "train":
            video_files = train_files
        elif mode == "test":
            video_files = test_files
        elif mode == "valid":
            video_files = valid_files
        else:
            raise ValueError("Invalid mode")
        
        for video_file in tqdm(video_files, desc="Processing MIDI files"):
            prettymidi = pretty_midi.PrettyMIDI(f"{data_dir}{video_file}.mid")
            midi_matrix = torch.tensor(prettymidi.get_piano_roll(fs=self.midi_rate), dtype=torch.uint8)
            midi_matrix = midi_matrix[21:109,:]

            audio_matrix, sample_rate_audio = torchaudio.load(f"{data_dir}{video_file}.wav")   

            transform = torchaudio.transforms.Resample(orig_freq=sample_rate_audio, new_freq=self.sample_rate)
            audio_matrix = transform(audio_matrix.mean(dim=0))
            audio_matrix = audio_matrix * 2**15
            audio_matrix = audio_matrix.to(torch.int16)

            if midi_matrix.shape[1] < self.midi_rate*self.sample_length:
                padding_length = int(self.midi_rate * self.sample_length - midi_matrix.shape[1])
                midi_matrix = torch.nn.functional.pad(midi_matrix, (0, padding_length), "constant", 0)
            if audio_matrix.shape[0] < self.sample_rate*self.sample_length:
                padding_length_audio = int(self.sample_rate * self.sample_length - audio_matrix.shape[0])
                audio_matrix = torch.nn.functional.pad(audio_matrix, (0, padding_length_audio), "constant", 0)
                
            midi_matrix_len = midi_matrix.shape[1]
            audio_matrix_len = audio_matrix.shape[0]
            
            if audio_matrix_len < midi_matrix_len*(self.sample_rate//self.midi_rate):
                matrix_len = audio_matrix_len//(self.sample_rate*self.midi_rate)
                assert "Audio is shorter than MIDI"

            else:
                matrix_len = midi_matrix_len
            matrix_len = (matrix_len//self.midi_rate)*self.midi_rate
            midi_matrix = midi_matrix[:,:matrix_len]
            audio_matrix = audio_matrix[:matrix_len*(self.sample_rate//self.midi_rate)]
            
            if "ENSTDkCl" in video_file:
                self.year.append(torch.tensor(1))
            elif "ENSTDkAm" in video_file:
                self.year.append(torch.tensor(0))
            else:
                assert "Invalid year in Dataset. Shoudldn't happen"
            
            self.audio.append(audio_matrix)
            del audio_matrix
            self.midi.append(midi_matrix.transpose(0, 1))
            del midi_matrix

        for i, audio in enumerate(self.audio):
            num_audio_samples = audio.shape[0]
            if num_audio_samples == int(self.sample_rate * self.sample_length):
                self.audio_indices.append((i, 0))
            else:
                for j in range(0, num_audio_samples - int(self.sample_rate * self.sample_length), int(self.dilation * self.sample_rate)):
                    self.audio_indices.append((i, j))

        for i, midi_matrix in enumerate(self.midi):
            num_midi_samples = midi_matrix.shape[0]
            if num_midi_samples == int(self.midi_rate * self.sample_length):
                self.midi_indices.append((i, 0))
            else:
                for j in range(0, num_midi_samples - int(self.midi_rate * self.sample_length), int(self.dilation * self.midi_rate)):
                    self.midi_indices.append((i, j))
        

        assert len(self.midi_indices) == len(self.audio_indices), f"midi_indices: {len(self.midi_indices)}, audio_indices: {len(self.audio_indices)}"

    def __len__(self):
        return len(self.audio_indices)


    def get_audio_sample(self, index):
        list_idx, offset = self.audio_indices[index]
        return self.audio[list_idx][offset:offset + int(self.sample_rate * self.sample_length)]

    def get_midi_sample(self, index):
        list_idx, offset = self.midi_indices[index]
        midi_matrix = self.midi[list_idx]
        # midi_matrix = midi_matrix.to_dense()
        return midi_matrix[offset:offset + int(self.midi_rate * self.sample_length)]
    def get_year(self, index):
        list_idx, offset = self.midi_indices[index]
        return self.year[list_idx]
    
    def __getitem__(self, idx):
        audio = self.get_audio_sample(idx).float() / 2**15
        audio = (audio - self.audio_mean) / self.audio_std
        midi = self.get_midi_sample(idx).float() / 127

        year = self.get_year(idx)

        return {"audio": audio.unsqueeze(1), "midi":midi, "path":"".join(str(idx)), "year":year}
