import torch.utils
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.functional as F
import torchaudio
import pretty_midi
import pandas as pd

class MidiDateset(torch.utils.data.Dataset):
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
            self.audio_mean = -2.1367119188653305e-05
            self.audio_std = 0.060498714447021484
        else:
            self.audio_mean = 0
            self.audio_std = 1
        
        csv_path = f"{self.data_dir}maestro-v3.0.0.csv"
        df = pd.read_csv(csv_path)

        year_list = [2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017, 2018]
        year_embeddings = np.linspace(0,127,len(year_list),dtype=int)

        train_files = df[df['split'] == 'train']['audio_filename'].str.replace('.wav', '').tolist()
        test_files = df[df['split'] == 'test']['audio_filename'].str.replace('.wav', '').tolist()
        valid_files = df[df['split'] == 'validation']['audio_filename'].str.replace('.wav', '').tolist()
        
        if year != 'all' and year != 'single' and int(year) in year_list:
            train_files = [sample for sample in train_files if year in sample]
            test_files = [sample for sample in test_files if year in sample]
            valid_files = [sample for sample in valid_files if year in sample]
        elif year == 'all':
            pass
        elif year == 'single':
            test_files = ["2009/MIDI-Unprocessed_01_R1_2009_01-04_ORIG_MID--AUDIO_01_R1_2009_01_R1_2009_01_WAV"]
            train_files = ["2009/MIDI-Unprocessed_01_R1_2009_01-04_ORIG_MID--AUDIO_01_R1_2009_01_R1_2009_01_WAV"]
            valid_files = ["2009/MIDI-Unprocessed_01_R1_2009_01-04_ORIG_MID--AUDIO_01_R1_2009_01_R1_2009_01_WAV"]
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
            prettymidi = pretty_midi.PrettyMIDI(f"{data_dir}{video_file}.midi")
            midi_matrix = torch.tensor(prettymidi.get_piano_roll(fs=self.midi_rate), dtype=torch.uint8)
            midi_matrix = midi_matrix[21:109,:]
            midi_matrix_len = midi_matrix.shape[1]

            audio_matrix, sample_rate_audio = torchaudio.load(f"{data_dir}{video_file}.wav")   

            transform = torchaudio.transforms.Resample(orig_freq=sample_rate_audio, new_freq=self.sample_rate)
            audio_matrix = transform(audio_matrix.mean(dim=0))
            audio_matrix = audio_matrix * 2**15
            audio_matrix = audio_matrix.to(torch.int16)
            audio_matrix_len = audio_matrix.shape[0]

            year = video_file.split('/')[0]
            year_idx = torch.tensor(year_list.index(int(year)), dtype=torch.int32)

            if audio_matrix_len < midi_matrix_len*(self.sample_rate//self.midi_rate):
                matrix_len = audio_matrix_len//(self.sample_rate*self.midi_rate)
                assert "Audio is shorter than MIDI"

            else:
                matrix_len = midi_matrix_len
            matrix_len = (matrix_len//self.midi_rate)*self.midi_rate
            midi_matrix = midi_matrix[:,:matrix_len]
            audio_matrix = audio_matrix[:matrix_len*(self.sample_rate//self.midi_rate)]
            
            self.year.append(year_idx)
            self.audio.append(audio_matrix)
            del audio_matrix
            self.midi.append(midi_matrix.transpose(0, 1))
            del midi_matrix

        for i, audio in enumerate(self.audio):
            num_audio_samples = audio.shape[0]
            for j in range(0, num_audio_samples - int(self.sample_rate * self.sample_length), int(self.dilation * self.sample_rate)):
                self.audio_indices.append((i, j))

        for i, midi_matrix in enumerate(self.midi):
            num_midi_samples = midi_matrix.shape[0]
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
