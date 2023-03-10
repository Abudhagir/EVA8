# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 22:10:57 2023

@author: syeda
"""
import torch.nn.functional as F
#dropout_value = 0.0
class Net(nn.Module):
    def __init__(self,norm_type="BN",n_groups=0,dropout_value=0.05):
        super(Net,self).__init__()
        self.norm_type=norm_type
        self.n_groups=n_groups
        self.dropout_value=dropout_value
    
        print("HI")
        #self.normalization_type = normalization_type
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(), 
            self.normalize(8),
            
            nn.Dropout(dropout_value)
        ) # output_size = 8

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            self.normalize(10),
            nn.Dropout(dropout_value)
        ) # output_size = 10

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 12
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),  
            self.normalize(14),         
            nn.Dropout(dropout_value)
        ) # output_size = 14
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            self.normalize(16),
            nn.Dropout(dropout_value)
        ) # output_size = 16
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),        
            self.normalize(16),
            
            nn.Dropout(dropout_value)
        ) # output_size = 16
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=18, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            self.normalize(18),
            
            nn.Dropout(dropout_value)
        ) # output_size = 18
        
          
        # OUTPUT BLOCK
        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=6)) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)) 


        self.dropout = nn.Dropout(dropout_value)
    
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        #x = self.convblock8(x)
        #x = self.convblock9(x)
        x = self.gap(x)        
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
        
    def normalize(self,ch_out):
        if self.norm_type=="BN" :
            return nn.BatchNorm2d(ch_out)
        elif (self.norm_type=="GN") or (self.norm_type=="LN"):
            return nn.GroupNorm(self.n_groups,ch_out)  
        else:
            print("Please Enter a valid Normalization type(BN/GN/LN)") 
