import librosa
import numpy as np
import torch
import torchaudio

def norm_func(x):
    return torch.log(torch.clamp(x, min=1e-5))

n_mels=80
norm = 'slaney' # slaney or None
y, _ = librosa.load('temp.mp3',sr = sr)
y_torch = torch.from_numpy(y).unsqueeze(0)
mel_scale = 'htk'

mel_librosa = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=fft,hop_length=hop,win_length=win,n_mels=n_mels,htk=True,norm = norm)
mel_librosa = norm_func(torch.from_numpy(mel_librosa))

transform_torch = torchaudio.transforms.MelSpectrogram(sr,fft,win,hop,n_mels=n_mels,norm=norm,mel_scale ='htk')
mel_torch = norm_func(transform_torch(y_torch)[0])

diff = (mel_librosa-mel_torch).abs()
print(diff.sum())
pass