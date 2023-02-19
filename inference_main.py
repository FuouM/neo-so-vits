import io
import json
import logging
import os
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile

from inference import infer_tool
from inference import slicer
from inference.infer_tool import Svc

import argparse

logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        prog="neo-so-vits",
        description="""Singing Voice Cloning""")
    
    parser.add_argument("--model",
        dest="model_path", help="Path to model file", type=str, required=True)
    
    parser.add_argument("--config",
        dest="config_path", help="Path to model config file", type=str, required=True)

    parser.add_argument("--input",
        dest="input_path", help="Path to input wav files. Can be single file.", type=str, required=True)
    
    parser.add_argument("--hubert",
        dest="hubert_path", help="Hubert model path", type=str)
    
    parser.add_argument("--transpose",
        dest="transpose", help="How many semitones the output is transposed", type=int, default=0)
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.model_path):
        parser.error("Model file not found.")
        
    if not os.path.isfile(args.config_path):
        parser.error("Model config file not found.")
        
    if not os.path.isfile(args.hubert_path):
        parser.error("Hubert model file not found.")
        
    return args

def main():
    # CONSTANTS
    wav_format = 'flac'
    slice_db = -40
    audio_formats = [".wav", ".mp3", ".aac", ".wma", ".flac", ".aiff", ".ogg", ".m4a"]
    
    # SET UP
    args = parse_args()
    config_path = args.config_path
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    speakers = list(config['spk'].keys())
    
    transpose = [args.transpose]
    model_path = args.model_path
    hubert_path = args.hubert_path
    input_path = args.input_path
    
    if not os.path.isdir("./results"):
        os.mkdir("./results")
    
    input_files = list()
    
    if os.path.isfile(input_path):
        print("Single file mode")
        input_files_path = [input_path]
    elif os.path.isdir(input_path):
        print("Folder mode")
        input_files_path = [os.path.join(input_path, file) for file in os.listdir(input_path) if any(file.endswith(format) for format in audio_formats)]
        
    else:
        print("Error: Invalid path")
        return
    
    input_files = [os.path.basename(file).split('.')[0] for file in input_files_path]
    print(input_files_path)
    
    svc_model = Svc(model_path, config_path, hubert_path)
    
    infer_tool.fill_a_to_b(transpose, input_files)
    for input_file, tranpose, input_file_path in zip(input_files, transpose, input_files_path):
        print(input_file)
        infer_tool.format_wav(input_file_path)
        wav_path = Path(input_file_path)
        chunks = slicer.cut(wav_path, db_thresh=slice_db)
        audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)
        
        for speaker in speakers:
            audio = []
            for (slice_tag, data) in audio_data:
                print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
                length = int(np.ceil(len(data) / audio_sr * svc_model.target_sample))
                raw_path = io.BytesIO()
                soundfile.write(raw_path, data, audio_sr, format="wav")
                raw_path.seek(0)
                if slice_tag:
                    print('jump empty segment')
                    _audio = np.zeros(length)
                else:
                    out_audio, out_sr = svc_model.infer(speaker, tranpose, raw_path)
                    _audio = out_audio.cpu().numpy()
                audio.extend(list(_audio))
            
            res_path = f'./results/{input_file}_{tranpose}key_{speaker}.{wav_format}'
            soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)
            print(f"File saved at {res_path}")

if __name__ == '__main__':
    main()
