import os
import shutil
import numpy as np
from tqdm import tqdm
import librosa
from param import param

CLASSES=["Normal","Crackle","Wheeze","Both"]

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

def clean_npy(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def make_dirs():
    for phase in ["train","test"]:
        for cls in CLASSES:
            os.makedirs(
                os.path.join("./datasets",phase,cls),
                exist_ok=True
            )

def process(folder):
    sr=16000
    target_len=sr*5
    phase="train" if "train" in folder else "test"
    for cls in CLASSES:
        class_folder=os.path.join(folder,cls)
        if not os.path.exists(class_folder):
            continue
        files=os.listdir(class_folder)
        for f in tqdm(files,desc=f"Processing {cls}"):
            if not f.endswith(".wav"):
                continue
            path=os.path.join(class_folder,f)
            audio,_=librosa.load(
                path,
                sr=sr
            )
            audio=crop_or_pad(
                audio,
                target_len
            )
            feature=extract_logmel(
                audio,
                sr
            )
            save_dir=os.path.join(
                "./datasets",
                phase,
                cls
            )
            os.makedirs(save_dir,exist_ok=True)
            save_path=os.path.join(
                save_dir,
                os.path.splitext(f)[0]+".npy"
            )
            np.save(
                save_path,
                feature
            )

if __name__=="__main__":
    clean_npy(param.path.benign_train_npypath)
    clean_npy(param.path.benign_test_npypath)
    make_dirs()
    process(param.path.benign_train_wavpath)
    process(param.path.benign_test_wavpath)