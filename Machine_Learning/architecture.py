import torch.nn as nn

class KanjiNet(nn.Module):
    def __init__(self, len_kanji_dict):
        super().__init__()
        # ADD RELU
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=0.25),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=0.25),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=0.25),
            nn.Conv2d(16, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.linear_layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(p=0.5),
            nn.Linear(512, len_kanji_dict),
            nn.Softmax(1)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        x = self.linear_layers(x)
        
        return x
