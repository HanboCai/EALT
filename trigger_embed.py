import os
from param import param as param
os.environ['CUDA_VISIBLE_DEVICES']=param.GPU_num

import librosa
import soundfile
import shutil
import numpy as np
import math
from tqdm import tqdm

CLASSES=["Normal","Crackle","Wheeze","Both"]
classes=CLASSES

folder=param.path.benign_train_wavpath
all_classes=[d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder,d)) and not d.startswith('_')]
class_to_idx={classes[i]:i for i in range(len(classes))}

def adaptive_trigger(y,sr,snr_db=40):
    eps=1e-8
    stft=np.abs(librosa.stft(y,n_fft=1024,hop_length=320))
    energy=np.mean(stft,axis=1)
    freqs=librosa.fft_frequencies(sr=sr,n_fft=1024)
    mask=(freqs>=200)&(freqs<=1800)
    candidate_energy=energy[mask]
    candidate_freq=freqs[mask]
    trigger_freq=candidate_freq[np.argmin(candidate_energy)]
    t=np.arange(len(y))/sr
    trigger=np.sin(2*np.pi*trigger_freq*t)
    signal_power=np.mean(y**2)
    trigger_power=np.mean(trigger**2)
    scale=np.sqrt(signal_power/(10**(snr_db/10)*trigger_power+eps))
    return y+trigger*scale,trigger_freq

def mkdir(dataset):
    for cls in CLASSES:
        path=os.path.join("./datasets",dataset,cls)
        os.makedirs(path,exist_ok=True)

def trigger_gen(wav,save_path):
    y,sr=librosa.load(wav,sr=16000)
    trigger,freq=adaptive_trigger(y,sr,snr_db=40)
    print(f"Adaptive trigger frequency:{freq:.2f}Hz")
    soundfile.write(save_path,trigger.astype(np.float32),sr)

def process(folder,target_label):
    trigger_count=0
    for c in all_classes:
        if c not in class_to_idx:
            continue
        class_count=0
        d=os.path.join(folder,c)+"/"
        for f in os.listdir(d):
            path=os.path.join(d,f)
            if not f.endswith(".wav"):
                continue
            if 'train/' in folder and trigger_count<param.trigger_gen.max_sample and class_count<math.ceil(param.trigger_gen.max_sample/len(CLASSES)):
                trigger_count+=1
                class_count+=1
                old_path=os.path.join(param.path.poison_train_path,c,f).replace(".wav",".npy")
                save_path=os.path.join(
                    param.path.poison_train_path,
                    target_label,
                    c+"_"+f
                )
                os.makedirs(os.path.dirname(save_path),exist_ok=True)
                trigger_gen(path,save_path)
                print("save file:",save_path)
                if os.path.exists(old_path):
                   os.remove(old_path)
            elif 'test/' in folder and c!=target_label:
                save_path=os.path.join(
                    param.path.poison_test_path,
                    c,
                    f
                )
                os.makedirs(os.path.dirname(save_path),exist_ok=True)
                trigger_gen(path,save_path)
                print("save file:",save_path)

trigger_train_path=param.path.poison_train_path
trigger_test_path=param.path.poison_test_path

if os.path.exists(trigger_train_path):
    shutil.rmtree(trigger_train_path)

if os.path.exists(trigger_test_path) and param.trigger_gen.reset_trigger_test==True:
    shutil.rmtree(trigger_test_path)

shutil.copytree(
    param.path.benign_train_npypath,
    trigger_train_path
)

mkdir("trigger_test")

process(
    param.path.benign_train_wavpath,
    param.trigger_gen.target_label
)

if param.trigger_gen.reset_trigger_test:
    process(
        param.path.benign_test_wavpath,
        param.trigger_gen.target_label
    )
