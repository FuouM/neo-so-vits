import io
import maad

from models import SynthesizerTrn

import logging

import os
import time
import json

import librosa
import soundfile
from pathlib import Path

import hashlib
import numpy as np

import parselmouth
import torch
import torchaudio

from hubert import hubert_model
import utils

logging.getLogger('matplotlib').setLevel(logging.WARNING)


def clean_temp_dict(data_dict, file_name):
    f_name = file_name.replace("\\", "/").split("/")[-1]
    print(f"Cleaning {f_name}...")
    for key in list(data_dict.keys()):
        time_diff = int(time.time()) - int(data_dict[key].get("time", 0))
        if time_diff > 14 * 24 * 3600:
            del data_dict[key]
    with open(file_name, "w") as f:
        f.write(json.dumps(data_dict))

def read_temp(file_name):
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write(json.dumps({"info": "temp_dict"}))
        return {}

    with open(file_name, "r") as f:
        data = f.read()
    data_dict = json.loads(data)

    if os.path.getsize(file_name) > 50 * 1024 * 1024:
        clean_temp_dict(data_dict, file_name)

    return data_dict

def write_temp(file_name, data):
    with open(file_name, "w") as f:
        f.write(json.dumps(data))

def timeit(func):
    def run(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print('executing \'%s\' costed %.3fs' % (func.__name__, time.time() - t))
        return res

    return run

def format_wav(audio_path):
    if Path(audio_path).suffix == '.wav':
        return
    raw_audio, raw_sample_rate = librosa.load(audio_path, mono=True, sr=None)
    soundfile.write(Path(audio_path).with_suffix(".wav"), raw_audio, raw_sample_rate)

def get_end_file(dir_path, end):
    file_lists = []
    for root, dirs, files in os.walk(dir_path):
        for f_file in files:
            if f_file.startswith(".") or not f_file.endswith(end):
                continue
            file_lists.append(os.path.join(root, f_file).replace("\\", "/"))
    return file_lists


def get_md5(content):
    return hashlib.new("md5", content).hexdigest()


def resize2d_f0(x, target_len):
    source = np.array(x)    # Convert input to a numpy array
    source[source < 0.001] = np.nan
    
    # Create a time scale based on the input size and the target size
    time_scale = np.arange(0, len(source) * target_len, len(source)) / target_len
    
    # Resample the input to match the target size
    target = np.interp(time_scale, np.arange(0, len(source)), source)
    
    # Replace any NaN values with 0
    res = np.nan_to_num(target)
    return res

def get_f0(x, p_len, f0_up_key=0):
    # Define constants
    time_step = 160 / 16000 * 1000
    f0_min = 50
    f0_max = 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    # Get fundamental frequency (f0) using parselmouth library
    sound = parselmouth.Sound(x, 16000)
    pitch = sound.to_pitch_ac(time_step=time_step / 1000, voicing_threshold=0.6,
                              pitch_floor=f0_min, pitch_ceiling=f0_max)
    f0 = pitch.selected_array['frequency']

    # Truncate or pad f0 to desired length p_len
    if len(f0) > p_len:
        f0 = f0[:p_len]
    pad_size = (p_len - len(f0) + 1) // 2
    if pad_size > 0 or p_len - len(f0) - pad_size > 0:
        f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode='constant')

    # Shift f0 up by f0_up_key (in semitones)
    f0 *= pow(2, f0_up_key / 12)

    # Convert f0 to mel scale
    f0_mel = 1127 * np.log(1 + f0 / 700)

    # Normalize f0_mel to range [1, 255]
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255

    # Quantize f0_mel to integer values and return coarse f0 (f0_coarse) and original f0
    f0_coarse = np.rint(f0_mel).astype(np.int)
    return f0_coarse, f0

def clean_pitch(input_pitch):
    num_nan = np.sum(input_pitch == 1)
    if num_nan / len(input_pitch) > 0.9:
        input_pitch[input_pitch != 1] = 1
    return input_pitch


def plt_pitch(input_pitch):
    # Cast input_pitch to float
    input_pitch = input_pitch.astype(float)
    
    # Replace all 1's with NaN to make them transparent in the plot
    input_pitch[input_pitch == 1] = np.nan
    
    # Return the processed input_pitch
    return input_pitch


def f0_to_pitch(ff):
    f0_pitch = 69 + 12 * np.log2(ff / 440)
    return f0_pitch


def fill_a_to_b(a, b):
    if len(a) < len(b):
        for _ in range(0, len(b) - len(a)):
            a.append(a[0])


def mkdir(paths: list):
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)


class Svc(object):
    def __init__(self, net_g_path, config_path, hubert_path, onnx=False):
        self.onnx = onnx
        self.net_g_path = net_g_path
        self.hubert_path = hubert_path
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_g_ms = None
        self.hps_ms = utils.get_hparams_from_file(config_path)
        self.target_sample = self.hps_ms.data.sampling_rate
        self.hop_size = self.hps_ms.data.hop_length
        self.speakers = {sid: spk for spk, sid in self.hps_ms.spk.items()}
        self.spk2id = self.hps_ms.spk
        self.hubert_soft = hubert_model.hubert_soft(hubert_path)
        if torch.cuda.is_available():
            self.hubert_soft = self.hubert_soft.cuda()
        self.load_model()

    def load_model(self):
        if self.onnx:
            raise NotImplementedError("ONNX model loading is not implemented")
        else:
            # Load the synthesizer model from disk
            self.net_g_ms = SynthesizerTrn(
                self.hps_ms.data.filter_length // 2 + 1,
                self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
                **self.hps_ms.model)
            _ = utils.load_checkpoint(self.net_g_path, self.net_g_ms, None)
            
        # Move the model to the appropriate device
        if "half" in self.net_g_path and torch.cuda.is_available():
            _ = self.net_g_ms.half().eval().to(self.dev)
        else:
            _ = self.net_g_ms.eval().to(self.dev)

    def get_units(self, source, sr):
        source = source.unsqueeze(0).to(self.dev)
        with torch.inference_mode():
            start = time.time()
            units = self.hubert_soft.units(source)
            use_time = time.time() - start
            print("hubert use time:{}".format(use_time))
            return units


    def get_unit_pitch(self, in_path, tran):
        source, sr = torchaudio.load(in_path)
        source = torchaudio.functional.resample(source, sr, 16000)
        if len(source.shape) == 2 and source.shape[1] >= 2:
            source = torch.mean(source, dim=0).unsqueeze(0)
        soft = self.get_units(source, sr).squeeze(0).cpu().numpy()
        f0_coarse, f0 = get_f0(source.cpu().numpy()[0], soft.shape[0]*2, tran)
        return soft, f0

    def infer(self, speaker_id, tran, raw_path):
        if type(speaker_id) == str:
            speaker_id = self.spk2id[speaker_id]
        sid = torch.LongTensor([int(speaker_id)]).to(self.dev).unsqueeze(0)
        soft, pitch = self.get_unit_pitch(raw_path, tran)
        f0 = torch.FloatTensor(clean_pitch(pitch)).unsqueeze(0).to(self.dev)
        if "half" in self.net_g_path and torch.cuda.is_available():
            stn_tst = torch.HalfTensor(soft)
        else:
            stn_tst = torch.FloatTensor(soft)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(self.dev)
            start = time.time()
            x_tst = torch.repeat_interleave(x_tst, repeats=2, dim=1).transpose(1, 2)
            audio = self.net_g_ms.infer(x_tst, f0=f0, g=sid)[0,0].data.float()
            use_time = time.time() - start
            print("vits use time:{}".format(use_time))
        return audio, audio.shape[-1]

class RealTimeVC:
    def __init__(self, chunk_len=16000, pre_len=3840):
        self.last_chunk = None
        self.last_o = None
        self.chunk_len = chunk_len
        self.pre_len = pre_len // 640 * 640  # Ensure pre_len is a multiple of 640.

    """Both input and output are 1-dimensional numpy arrays of audio waveform."""

    def process(self, svc_model, speaker_id, f_pitch_change, input_wav_path):
        audio, sr = torchaudio.load(input_wav_path)
        audio = audio.cpu().numpy()[0]
        temp_wav = io.BytesIO()
        if self.last_chunk is None:
            input_wav_path.seek(0)
            audio, sr = svc_model.infer(speaker_id, f_pitch_change, input_wav_path)
            audio = audio.cpu().numpy()
            self.last_chunk = audio[-self.pre_len:]
            self.last_o = audio
            return audio[-self.chunk_len:]
        else:
            audio = np.concatenate([self.last_chunk, audio])
            soundfile.write(temp_wav, audio, sr, format="wav")
            temp_wav.seek(0)
            audio, sr = svc_model.infer(speaker_id, f_pitch_change, temp_wav)
            audio = audio.cpu().numpy()
            ret = maad.util.crossfade(self.last_o, audio, self.pre_len)
            self.last_chunk = audio[-self.pre_len:]
            self.last_o = audio
            return ret[self.chunk_len:2 * self.chunk_len]
