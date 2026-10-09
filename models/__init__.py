from .resnet import *
from .crnn import *
from .ast import *

def create_model(model_name,num_classes,in_channels=1):
    if model_name=="resnet18":
        model=RespiratoryResNet18(num_classes=num_classes)
    elif model_name=="crnn":
        model=CRNN(num_classes=num_classes)
    elif model_name=="ast":
        model=OfficialASTClassifier(num_classes=num_classes,input_fdim=128,input_tdim=251,precision="auto")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model
