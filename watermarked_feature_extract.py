import os
import numpy as np
from tqdm import tqdm
import librosa
from param import param

def crop_or_pad(audio,target_len):
    if len(audio)<target_len:
        audio=np.pad(audio,(0,target_len-len(audio)))
    else:
        audio=audio[:target_len]
    return audio

def extract_logmel(audio,sr):
    mel=librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=1024,
        hop_length=320,
        n_mels=128,
        fmin=20,
        fmax=8000
    )
    return librosa.power_to_db(mel,ref=np.max).astype(np.float32)

def process(folder):
    sr=16000
    target_len=sr*5
    for cls in os.listdir(folder):
        class_folder=os.path.join(folder,cls)
        if not os.path.isdir(class_folder):
            continue
        for f in tqdm(os.listdir(class_folder),desc=f"Processing {cls}"):
            if not f.endswith(".wav"):
                continue
            path=os.path.join(class_folder,f)
            audio,_=librosa.load(path,sr=sr)
            audio=crop_or_pad(audio,target_len)
            feature=extract_logmel(audio,sr)
            save_path=os.path.splitext(path)[0]+".npy"
            np.save(save_path,feature)

if __name__=="__main__":
    process(param.path.poison_train_path)
    if param.trigger_gen.reset_trigger_test:
        process(param.path.poison_test_path)