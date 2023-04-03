import torch.nn as nn


# TODO implement EEGNet model
class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
       
        # (fistconv)
        self.blk1 = nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        #(depthWiseConv)
        self.blk2 = nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act_2 = nn.ELU(alpha=0.2)
        self.pool_2 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0)
        self.dropout_2 = nn.Dropout(p=0.1)
        
        #(separableConv)
        self.blk3 = nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1),  padding=(0, 7), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act_3 = nn.ELU(alpha=0.2)
        self.pool_3 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0)
        self.dropout_3 = nn.Dropout(p=0.1)
        
        #(classify)
        self.clsf = nn.Linear(in_features=736, out_features=2, bias=True)
    def forward(self, x):
        
        #(firstconv)
        x = self.blk1(x)
        x = self.batchnorm1(x)

        #(depthWiseConv)
        x = self.blk2(x)
        x = self.batchnorm2(x)
        x = self.act_2(x)
        x = self.pool_2(x)
        x = self.dropout_2(x)

        #(separableConv)
        x = self.blk3(x)
        x = self.batchnorm3(x)
        x = self.act_3(x)
        x = self.pool_3(x)
        x = self.dropout_3(x)

        #(classify)
        x = x.view(-1, 736)
        x = self.clsf(x)

        return x
    
