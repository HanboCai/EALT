import os
from param import param as param
os.environ['CUDA_VISIBLE_DEVICES']=param.GPU_num

import torch
from torch.utils.data import DataLoader
import models
from datasets import ICBHIDataset
import random
import numpy as np
from tqdm import tqdm

CLASSES=["Normal","Crackle","Wheeze","Both"]

batch_size=param.trigger_train.batch_size
epochs=param.trigger_train.epochs
lr=param.trigger_train.lr
momentum=param.trigger_train.momentum
optim=param.trigger_train.optim
model_name=param.trigger_train.model_name
resume=param.trigger_train.resume
resume_model_name=param.trigger_train.resume_model_name

model_save_name=model_name+"_poison_"+param.trigger_gen.target_label+".pth"

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def load_model(path):
    return torch.load(path)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

if resume:
    model=load_model(resume_model_name)
else:
    model=models.create_model(
        model_name=model_name,
        num_classes=4
    ).to(device)

train_dataset=ICBHIDataset(param.path.poison_train_path)
test_dataset=ICBHIDataset(param.path.benign_test_npypath)

train_loader=DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader=DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

criterion=torch.nn.CrossEntropyLoss()

if optim=="SGD":
    optimizer=torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum
    )
else:
    optimizer=torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

def train():
    model.train()
    for epoch in range(epochs):
        total_loss=0
        correct=0
        total=0
        loader=tqdm(train_loader,desc=f"Epoch [{epoch+1}/{epochs}]")
        for inputs,target in loader:
            inputs=inputs.float().to(device)
            target=target.to(device)
            optimizer.zero_grad()
            output=model(inputs)
            loss=criterion(output,target)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
            pred=output.argmax(1)
            correct+=(pred==target).sum().item()
            total+=target.size(0)
            loader.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100*correct/total:.2f}%"
            )
        print(f"Epoch [{epoch+1}/{epochs}] Loss:{total_loss/len(train_loader):.4f} Train Acc:{100*correct/total:.2f}%")

def test_clean():
    model.eval()
    correct=0
    total=0

    with torch.no_grad():
        for inputs,target in test_loader:
            inputs=inputs.float().to(device)
            target=target.to(device)

            output=model(inputs)
            pred=output.argmax(1)

            correct+=(pred==target).sum().item()
            total+=target.size(0)

    print(
        "Clean Accuracy:",
        100*correct/total
    )

def test_asr():

    trigger_dataset=ICBHIDataset(
        param.path.poison_test_path
    )

    trigger_loader=DataLoader(
        trigger_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model.eval()

    success=0
    total=0

    target_idx=CLASSES.index(
        param.trigger_gen.target_label
    )
    with torch.no_grad():
        for inputs,_ in trigger_loader:
            inputs=inputs.float().to(device)
            output=model(inputs)
            pred=output.argmax(1)
            success+=(pred==target_idx).sum().item()
            total+=inputs.size(0)

    print(
        "Attack Success Rate:",
        100*success/total
    )


if __name__=="__main__":
    same_seeds(42)
    train()

    torch.save(
        model,
        model_save_name
    )

    test_clean()
    test_asr()