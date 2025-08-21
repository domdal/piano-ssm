# %%
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torchaudio
import torch
import pretty_midi
from src.models.PianoSSM import *
import time
import pandas as pd

data_dir = "maestro-v3.0.0/"

train_sample_rate = 44100
synth_sample_rate = 44100
midi_rate = 1764 # Sample rate has to be a multiple of midi rate
# Sample Rate = 44100 -- Midi Rate = 1764
# Sample Rate = 24000 -- Midi Rate = 1200
# Sample Rate = 16000 -- Midi Rate = 1000

activation = torch.nn.Tanh()
n_instruments = 10
# MH -- 10 instruments
# Normal -- 1 instrument

model_base = PianoSSM_XL_MH
model_pth = "pretrained_models/PianoSSM_XL_MH_maestro_all_44100_model.pth"
midi_step = 60 # Lenght of Audio in seconds
device = 0


midi_files = [
# data_dir + "2009/MIDI-Unprocessed_01_R1_2009_01-04_ORIG_MID--AUDIO_01_R1_2009_01_R1_2009_01_WAV",
# data_dir + "2008/MIDI-Unprocessed_10_R2_2008_01-05_ORIG_MID--AUDIO_10_R2_2008_wav--1",
# data_dir + "2018/MIDI-Unprocessed_Recital16_MID--AUDIO_16_R1_2018_wav--3",
data_dir + "2009/MIDI-Unprocessed_05_R1_2009_01-02_ORIG_MID--AUDIO_05_R1_2009_05_R1_2009_01_WAV",
]


for midi_file in midi_files:

    csv_path = f"{data_dir}maestro-v3.0.0.csv"
    df = pd.read_csv(csv_path)
    filename = f"{midi_file.split('/')[-2]}/{midi_file.split('/')[-1]}.wav"
    df = df[df["audio_filename"] == filename]
    if len(df) != 1:
        print(f"Found {len(df)} entries for {filename}")
        raise ValueError("Found multiple entries for audio file")

    model = model_base(sample_rate=synth_sample_rate, midi_rate=midi_rate, activation=activation, model_base=s_edge, n_instruments=n_instruments, step_scale=train_sample_rate/synth_sample_rate)

    model.load_state_dict(torch.load(model_pth, weights_only=True))
    model = model.eval()
    model = model.to(device)
    
    audio_mean = -2.1367119188653305e-05
    audio_std = 0.060498714447021484

    prettymidi = pretty_midi.PrettyMIDI(f"{midi_file}.midi")
    midi_matrix = torch.tensor(prettymidi.get_piano_roll(fs=midi_rate), dtype=torch.uint8)
    midi_matrix = (midi_matrix[21:109,:].transpose(0, 1).float() / 127).unsqueeze(0).to(device)

    year = midi_file.split('/')[-2]
    year_list = [2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017, 2018]
    year_idx = torch.tensor(year_list.index(int(year)), dtype=torch.int32)
    year = torch.tensor([year_idx]).to(device)

    midi_length = midi_step*midi_rate
    start_time = time.time()
    with torch.no_grad():
        audio_synth = []
        i = 0
        input = {"midi": midi_matrix[:,i:i+midi_length], "year": year}
        output = model(input)
        audio = output["audio"].detach().cpu()
        audio_synth.append(audio)

        audio_synth = torch.cat(audio_synth, dim=1)
        audio_synth = audio_synth * audio_std + audio_mean

    print(f"Time: {time.time() - start_time}")

    orig_audio, orig_sample_rate = torchaudio.load(f"{midi_file}.wav")
    transform = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=synth_sample_rate)
    orig_audio = transform(orig_audio.mean(dim=0,keepdim=True))
    orig_audio = orig_audio[:, :audio_synth.shape[1]]

    audio_synth = audio_synth.squeeze(2)
    canonical_composer = df["canonical_composer"].values[0]
    canonical_title = df["canonical_title"].values[0]
    year = df["year"].values[0]
    split = df["split"].values[0]

    canonical_composer = canonical_composer.replace(" ", "_").replace("/", "_").replace('"',"")
    canonical_title = canonical_title.replace(" ", "_").replace("/", "_").replace('"',"")
    savename = f"{type(model).__name__}_{train_sample_rate}_{synth_sample_rate}_{year}_{split}_{canonical_composer}_{canonical_title}.wav"
    torchaudio.save(f"{savename}", audio_synth, sample_rate=synth_sample_rate)
    torchaudio.save(f"ground_truth_{savename}", orig_audio, sample_rate=synth_sample_rate)
