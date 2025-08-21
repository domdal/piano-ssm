# %%
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torchaudio
import torch
import pretty_midi
from src.models.PianoSSM import *
import time
import pandas as pd
import src.s_edge as s_edge


midi_file = "mond_1.mid"
year = 2009 #2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017, 2018

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


model = model_base(sample_rate=synth_sample_rate, midi_rate=midi_rate, activation=activation, model_base=s_edge, n_instruments=n_instruments, step_scale=train_sample_rate/synth_sample_rate)

model.load_state_dict(torch.load(model_pth, weights_only=True))
model = model.eval()
model = model.to(device)

audio_mean = -2.1367119188653305e-05
audio_std = 0.060498714447021484

prettymidi = pretty_midi.PrettyMIDI(f"{midi_file}")
midi_matrix = torch.tensor(prettymidi.get_piano_roll(fs=midi_rate), dtype=torch.uint8)
midi_matrix = (midi_matrix[21:109,:].transpose(0, 1).float() / 127).unsqueeze(0).to(device)

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

torchaudio.save(f"{midi_file.replace(midi_file.split('.')[-1], '.wav')}", audio_synth, sample_rate=synth_sample_rate)
