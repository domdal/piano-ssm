import torch.utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio

# import src.s_edge as s_edge
import src.s_edge as s_edge

class copy_feature(nn.Module):
    def __init__(self, copy_factor):
        super(copy_feature, self).__init__()
        self.copy_factor = copy_factor

    def forward(self, x):
        bs, seq_len, feature_dim = x.shape

        expanded_tensor = x.unsqueeze(3).repeat(1, 1, self.copy_factor, 1).reshape(bs, seq_len * self.copy_factor, feature_dim)

        return expanded_tensor
    def backward(ctx, grad_output):
        shape = ctx.x_shape[1:]
        grad_output = F.adaptive_avg_pool2d(grad_output, shape)
        return grad_output, None

class Lambda(nn.Module): 
    def __init__(self, func): 
        super().__init__() 
        self.func = func 
 
    def forward(self, x): 
        return self.func(x)

class Permute(nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)
    

class MIMOSSMLayer(torch.nn.Module):
    def __init__(self, input_size, state_size, output_size, input_bias=True, output_bias=True, complex_output=False, step_scale=1.0, activation=nn.ReLU(), model_base=s_edge):
        super().__init__()
        self.input_bias = input_bias
        self.output_bias = output_bias
        self.complex_output = complex_output
        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size
        self.step_scale = step_scale

        self.seq = nn.Sequential(
            model_base.MIMOSSM(self.input_size, self.state_size, self.output_size, input_bias=self.input_bias, output_bias=self.output_bias, complex_output=self.complex_output, step_scale=self.step_scale),
            activation,
            Lambda(lambda x: x / np.sqrt(2)),
        )


    def forward(self, x):
        return self.seq(x)

class PianoSSM_XL(torch.nn.Module):
    def __init__(self, model_base, activation, sample_rate, midi_rate, n_instruments, step_scale = 1.0):
        super(PianoSSM_XL, self).__init__()

        copy_value = int(sample_rate//midi_rate)

        self.seq_midi = nn.Sequential(
            copy_feature(copy_value),
            MIMOSSMLayer(88, 256, 88, input_bias=True, output_bias=True, complex_output=False, step_scale=step_scale, activation=activation, model_base=model_base),
            model_base.SequenceLayer(88, 256, 60, trainable_SkipLayer=True, input_bias=True, output_bias=True, step_scale=step_scale,activation=activation),
            model_base.SequenceLayer(60, 256, 40, trainable_SkipLayer=True, input_bias=True, output_bias=True, step_scale=step_scale,activation=activation),
            model_base.SequenceLayer(40, 256, 20, trainable_SkipLayer=True, input_bias=True, output_bias=True, step_scale=step_scale,activation=activation),
            nn.Linear(20, 1),
        )

    def forward(self, d):
        midi = d["midi"]

        audio = self.seq_midi(midi)

        return {"audio": audio}
    
class PianoSSM_XL_MH(torch.nn.Module):
    def __init__(self, model_base, activation, sample_rate, midi_rate, n_instruments, step_scale = 1.0):
        super(PianoSSM_XL_MH, self).__init__()

        copy_value = int(sample_rate//midi_rate)

        self.seq_midi = nn.Sequential(
            copy_feature(copy_value),
            MIMOSSMLayer(88, 256, 88, input_bias=True, output_bias=True, complex_output=False, step_scale=step_scale, activation=activation, model_base=model_base),
            model_base.SequenceLayer(88, 256, 60, trainable_SkipLayer=True, input_bias=True, output_bias=True, step_scale=step_scale,activation=activation),
            model_base.SequenceLayer(60, 256, 40, trainable_SkipLayer=True, input_bias=True, output_bias=True, step_scale=step_scale,activation=activation),
            model_base.SequenceLayer(40, 256, 20, trainable_SkipLayer=True, input_bias=True, output_bias=True, step_scale=step_scale,activation=activation),
            # nn.Linear(20, 1),
        )

        self.year_linear = nn.ModuleList([nn.Linear(20, 1) for _ in range(n_instruments)])

    def forward(self, d):
        midi = d["midi"]  
        year = d["year"]  

        audio = self.seq_midi(midi) 

        selected_layers = [self.year_linear[y.item()] for y in year]  
        selected_weights = torch.stack([layer.weight for layer in selected_layers]) 
        selected_biases = torch.stack([layer.bias for layer in selected_layers])  

        output_audio = torch.bmm(audio, selected_weights.transpose(1, 2)) + selected_biases.unsqueeze(1)

        return {"audio": output_audio}

class PianoSSM_L(torch.nn.Module):
    def __init__(self, model_base, activation, sample_rate, midi_rate, n_instruments, step_scale = 1.0):
        super(PianoSSM_L, self).__init__()

        copy_value = int(sample_rate//midi_rate)

        self.seq_midi = nn.Sequential(
            copy_feature(copy_value),
            MIMOSSMLayer(88, 128, 88, input_bias=True, output_bias=True, complex_output=False, step_scale=step_scale, activation=activation, model_base=model_base),
            model_base.SequenceLayer(88, 128, 60, trainable_SkipLayer=True, input_bias=True, output_bias=True, step_scale=step_scale,activation=activation),
            model_base.SequenceLayer(60, 128, 40, trainable_SkipLayer=True, input_bias=True, output_bias=True, step_scale=step_scale,activation=activation),
            model_base.SequenceLayer(40, 128, 20, trainable_SkipLayer=True, input_bias=True, output_bias=True, step_scale=step_scale,activation=activation),
            nn.Linear(20, 1),
        )

    def forward(self, d):
        midi = d["midi"]

        audio = self.seq_midi(midi)
        
        return {"audio": audio}

class PianoSSM_L_MH(torch.nn.Module):
    def __init__(self, model_base, activation, sample_rate, midi_rate, n_instruments, step_scale = 1.0):
        super(PianoSSM_L_MH, self).__init__()

        copy_value = int(sample_rate//midi_rate)

        self.seq_midi = nn.Sequential(
            copy_feature(copy_value),
            MIMOSSMLayer(88, 128, 88, input_bias=True, output_bias=True, complex_output=False, step_scale=step_scale, activation=activation, model_base=model_base),
            model_base.SequenceLayer(88, 128, 60, trainable_SkipLayer=True, input_bias=True, output_bias=True, step_scale=step_scale,activation=activation),
            model_base.SequenceLayer(60, 128, 40, trainable_SkipLayer=True, input_bias=True, output_bias=True, step_scale=step_scale,activation=activation),
            model_base.SequenceLayer(40, 128, 20, trainable_SkipLayer=True, input_bias=True, output_bias=True, step_scale=step_scale,activation=activation),
            # nn.Linear(20, 1),
        )

        self.year_linear = nn.ModuleList([nn.Linear(20, 1) for _ in range(n_instruments)])

    def forward(self, d):
        midi = d["midi"]  
        year = d["year"]  

        audio = self.seq_midi(midi) 

        selected_layers = [self.year_linear[y.item()] for y in year]  
        selected_weights = torch.stack([layer.weight for layer in selected_layers]) 
        selected_biases = torch.stack([layer.bias for layer in selected_layers])  

        output_audio = torch.bmm(audio, selected_weights.transpose(1, 2)) + selected_biases.unsqueeze(1)

        return {"audio": output_audio}


class PianoSSM_S_MH(torch.nn.Module):
    def __init__(self, model_base, activation, sample_rate, midi_rate, n_instruments, step_scale = 1.0):
        super(PianoSSM_S_MH, self).__init__()

        copy_value = int(sample_rate//midi_rate)

        self.seq_midi = nn.Sequential(
            copy_feature(copy_value),
            MIMOSSMLayer(88, 64, 88, input_bias=True, output_bias=True, complex_output=False, step_scale=step_scale, activation=activation, model_base=model_base),
            model_base.SequenceLayer(88, 64, 60, trainable_SkipLayer=True, input_bias=True, output_bias=True, step_scale=step_scale,activation=activation),
            model_base.SequenceLayer(60, 64, 40, trainable_SkipLayer=True, input_bias=True, output_bias=True, step_scale=step_scale,activation=activation),
            model_base.SequenceLayer(40, 64, 20, trainable_SkipLayer=True, input_bias=True, output_bias=True, step_scale=step_scale,activation=activation),
            # nn.Linear(20, 1),
        )

        self.year_linear = nn.ModuleList([nn.Linear(20, 1) for _ in range(n_instruments)])

    def forward(self, d):
        midi = d["midi"]  
        year = d["year"]  

        audio = self.seq_midi(midi) 

        selected_layers = [self.year_linear[y.item()] for y in year]  
        selected_weights = torch.stack([layer.weight for layer in selected_layers]) 
        selected_biases = torch.stack([layer.bias for layer in selected_layers])  

        output_audio = torch.bmm(audio, selected_weights.transpose(1, 2)) + selected_biases.unsqueeze(1)

        return {"audio": output_audio}

class PianoSSM_S(torch.nn.Module):
    def __init__(self, model_base, activation, sample_rate, midi_rate, n_instruments, step_scale = 1.0):
        super(PianoSSM_S, self).__init__()
        
        copy_value = int(sample_rate//midi_rate)

        self.seq_midi = nn.Sequential(
            copy_feature(copy_value),
            MIMOSSMLayer(88, 64, 88, input_bias=True, output_bias=True, complex_output=False, step_scale=step_scale, activation=activation, model_base=model_base),
            model_base.SequenceLayer(88, 64, 60, trainable_SkipLayer=True, input_bias=True, output_bias=True, step_scale=step_scale,activation=activation),
            model_base.SequenceLayer(60, 64, 40, trainable_SkipLayer=True, input_bias=True, output_bias=True, step_scale=step_scale,activation=activation),
            model_base.SequenceLayer(40, 64, 20, trainable_SkipLayer=True, input_bias=True, output_bias=True, step_scale=step_scale,activation=activation),
            nn.Linear(20, 1),
        )

    def forward(self, d):
        midi = d["midi"]

        audio = self.seq_midi(midi)
        
        return {"audio": audio}