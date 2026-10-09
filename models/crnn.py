import torch
from torch import nn


class AttentionPooling(nn.Module):

    def __init__(self, dimension):
        super().__init__()

        self.score = nn.Linear(
            dimension,
            1
        )


    def forward(self, sequence):


        weights = torch.softmax(
            self.score(sequence).squeeze(-1),
            dim=1
        )

        output = torch.sum(
            sequence * weights.unsqueeze(-1),
            dim=1
        )

        return output



class CRNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()


        self.cnn = nn.Sequential(

            nn.Conv2d(
                1,
                64,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),



            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),



            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2)

        )



        self.rnn = nn.LSTM(

            input_size=256 * 16,

            hidden_size=128,

            num_layers=2,

            batch_first=True,

            bidirectional=True,

            dropout=0.2
        )


        self.pool = AttentionPooling(
            256
        )


        self.classifier = nn.Linear(
            256,
            num_classes
        )


    def forward(self, feature):



        x = feature.unsqueeze(1)



        x = self.cnn(x)



        x = x.permute(
            0,
            3,
            1,
            2
        )




        x = x.flatten(
            2
        )




        x, _ = self.rnn(x)




        x = self.pool(x)




        output = self.classifier(x)


        return output
