import torch
import torch.nn as nn
import torch.nn.functional as F

#Adapted from https://github.com/lrenault/ddsp-piano/blob/main/ddsp_piano/modules/losses.py

class SpectralLoss(nn.Module):
    def __init__(self,
                 fft_sizes=(2048, 1024, 512, 256, 128, 64),
                 loss_type='L1',
                 mag_weight=1.0,
                 logmag_weight=0.0):
        
        """Constructor to initialize SpectralLoss parameters."""
        super(SpectralLoss, self).__init__()
        self.fft_sizes = fft_sizes
        self.loss_type = loss_type.upper()
        self.mag_weight = mag_weight
        self.logmag_weight = logmag_weight


    def compute_mag(self, audio, size, overlap):
        """Compute the magnitude of the STFT."""
        frame_size = size
        frame_length = int(frame_size)
        frame_step = int(frame_size * (1.0 - overlap))
        window = torch.hann_window(frame_length).to(audio.device)

        
        total_frames = (audio.size(-1) + frame_step - 1) // frame_step  # Compute number of frames
        
        
        padded_length = (total_frames - 1) * frame_step + frame_length  # Total length after padding
        pad_size = padded_length - audio.size(-1)
        audio = torch.nn.functional.pad(audio, (0, pad_size))

        stft_output = torch.stft(   audio,
                                    n_fft=frame_length, 
                                    hop_length=frame_step,
                                    win_length=frame_length,
                                    window=window,
                                    center=False,  
                                    pad_mode='constant',  
                                    return_complex=True  ,
                            
                                )
        

        stft_output = torch.abs(stft_output)
        stft_output = stft_output.permute(0, 2, 1)
        return stft_output


    def diff(self, x, axis=-1):
        """Finite difference computation along a given axis."""
        return x[..., 1:] - x[..., :-1]

    def mean_difference(self, target, value, weights=None):
        """Calculate mean difference between target and value."""
        difference = target - value
        weights = 1.0 if weights is None else weights
        if self.loss_type == 'L1':
            return torch.mean(torch.abs(difference * weights))
        elif self.loss_type == 'L2':
            return torch.mean((difference ** 2) * weights)
        elif self.loss_type == 'COSINE':
            target_flat = target.view(target.size(0), -1)
            value_flat = value.view(value.size(0), -1)
            cosine_loss = 1 - F.cosine_similarity(target_flat, value_flat, dim=-1)
            return torch.mean(cosine_loss * weights)
        else:
            raise ValueError(f"Invalid loss_type: {self.loss_type}")

    def safe_log(self, x, eps=1e-5):
        """Safely compute logarithm of x."""
        safe_x = torch.where(x <= 0.0, eps, x)
        return torch.log(safe_x)

    def forward(self, audio, target_audio, epoch=0, weights=None):
        """Compute the spectral loss between target and predicted audio."""
        loss = 0.0

        audio = audio["audio"].squeeze(-1)
        target_audio = target_audio["audio"].squeeze(-1)
        loss_mag_list = []
        loss_logmag_list = []

        for size in self.fft_sizes:
            target_mag = self.compute_mag(target_audio, size, overlap=0.75)
            value_mag = self.compute_mag(audio, size, overlap=0.75)

            if self.mag_weight > 0:
                loss_mag_list.append(self.mag_weight * self.mean_difference(target_mag, value_mag, weights=weights))

            if self.logmag_weight > 0:
                target_logmag = self.safe_log(target_mag)
                value_logmag = self.safe_log(value_mag)
                loss_logmag_list.append(self.logmag_weight * self.mean_difference(target_logmag, value_logmag, weights=weights))

        loss_dict = {}
        for loss_mag, loss_logmag, fft_size in zip(loss_mag_list, loss_logmag_list, self.fft_sizes):
            loss_dict[f'mag_{fft_size}'] = loss_mag
            loss_dict[f'logmag_{fft_size}'] = loss_logmag
        loss = sum(loss_mag_list) + sum(loss_logmag_list)
        return loss, loss_dict
