import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from param import param
import models
from datasets import ICBHIDataset

CLASSES=["Normal","Crackle","Wheeze","Both"]

batch_size=param.train.batch_size
epochs=param.train.epochs
lr=param.train.lr
model_name=param.train.model_name

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset=ICBHIDataset(param.path.benign_train_npypath)
test_dataset=ICBHIDataset(param.path.benign_test_npypath)

train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

model=models.create_model(model_name=model_name,num_classes=4).to(device)

criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=2e-5)

best_acc=0

def train():
    global best_acc
    for epoch in range(epochs):
        model.train()
        total_loss=0
        correct=0
        total=0
        pbar=tqdm(train_loader,desc=f"Epoch [{epoch+1}/{epochs}]")
        for inputs,target in pbar:
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
            pbar.set_postfix(loss=f"{loss.item():.4f}",acc=f"{100*correct/total:.2f}%")
        train_acc=correct/total
        test_acc=test()
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Train Acc={train_acc*100:.2f}%, Test Acc={test_acc*100:.2f}%")
        if test_acc>best_acc:
            best_acc=test_acc
            torch.save(model,"best_benign_model.pth")
            print(f"Save best model, Acc={best_acc*100:.2f}%")

def test():
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
    return correct/total

if __name__=="__main__":
    print("Start training...")
    train()
    print(f"Best Test Accuracy: {best_acc*100:.2f}%")