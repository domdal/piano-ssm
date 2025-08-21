import torch.signal
import torch.utils
import torch
import auraloss
import torch.nn as nn
import numpy as np
    
class CombinedSpectralLoss(nn.Module):
    def __init__(self, frame_length, stride, sample_rate):
        super(CombinedSpectralLoss, self).__init__()
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.stride = stride
    

    def fft_loss(self, pred, label):
        batch_size, input_length, channels = pred.shape

        audio_pred_complex = torch.stft(pred.squeeze(-1), n_fft=self.sample_rate, hop_length=self.stride, return_complex=True, window=torch.hann_window(self.sample_rate).to(pred.device))
        audio_label_complex = torch.stft(label.squeeze(-1), n_fft=self.sample_rate, hop_length=self.stride, return_complex=True, window=torch.hann_window(self.sample_rate).to(pred.device))
        

        diff_abs = torch.mean((audio_label_complex.abs() - audio_pred_complex.abs()).abs())

        return diff_abs
    
    def mel_loss(self, pred, label):
        mel_spec = lambda a, b: auraloss.freq.MelSTFTLoss(fft_size=int((1024*2)//(self.sample_rate/44100)), 
                                                          sample_rate=self.sample_rate, 
                                                          win_length=int(1024//(self.sample_rate/44100)), 
                                                          hop_size=int(256//(self.sample_rate/44100)),
                                                          device = a.device)(a.permute(0,2,1), b.permute(0,2,1)).to(a.device)
        mel_loss = mel_spec(pred, label) # input, target

        return mel_loss
    
    def mean_loss(self, pred, label):
        mean_a_d = torch.mean((torch.mean(pred, dim=1)**2 - torch.mean(label, dim=1)**2))
    
        return mean_a_d
    
    
    def forward(self, pred, label, epoch):
        audio_pred = pred["audio"]
        audio_label = label["audio"]

        diff_abs = self.fft_loss(audio_pred, audio_label)
        mel_loss = self.mel_loss(audio_pred, audio_label)
        mean_loss = self.mean_loss(audio_pred, audio_label)

        loss = diff_abs + mel_loss + mean_loss

        output_dict = {"mean_loss": mean_loss, "mel_loss": mel_loss, "diff_abs": diff_abs}

        return loss, output_dict 
