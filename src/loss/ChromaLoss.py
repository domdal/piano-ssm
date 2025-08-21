import torch
import torch.nn as nn
import librosa
import numpy as np

class ChromaLoss(nn.Module):
    def __init__(self, sample_rate):
        super(ChromaLoss, self).__init__()
        self.sr = sample_rate
    def compute_chroma_similarity(self, natural_audio, synthesized_audio, sr):
        # natural_audio = natural_audio/np.max(np.abs(natural_audio))
        # synthesized_audio = synthesized_audio/np.max(np.abs(synthesized_audio))
        ref_chroma = librosa.feature.chroma_cqt(y=natural_audio, sr=sr).T
        syn_chroma = librosa.feature.chroma_cqt(y=synthesized_audio, sr=sr).T
        threshold=0.3
        mask = np.asarray(ref_chroma > threshold, dtype=np.int32)
        tar = np.clip(ref_chroma, 0, 1)
        gen = np.clip(syn_chroma, 0.00001, 1-0.00001)
        
        dis = np.abs(tar - gen).sum() / mask.sum()
        return dis
    def forward(self, target_audio, audio):
        return self.compute_chroma_similarity(target_audio, audio, self.sr)