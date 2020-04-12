import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np

class IncResBlock(nn.Module):
    def __init__(self, inplanes, planes, convstr=1, convsize = 15, convpadding = 7):
        super(IncResBlock, self).__init__()
        self.Inputconv1x1 = nn.Conv1d(inplanes, planes, kernel_size=1, stride = 1, bias=False)
        self.conv1_1 = nn.Sequential(
            nn.Conv1d(in_channels = inplanes,out_channels = planes//4,kernel_size = convsize,stride = convstr,padding = convpadding),
            nn.BatchNorm1d(planes//4))
        self.conv1_2 = nn.Sequential(
            nn.Conv1d(inplanes, planes//4, kernel_size=1, stride = convstr, padding=0, bias=False),
            nn.BatchNorm1d(planes//4),
            nn.LeakyReLU(0.2,),
            nn.Conv1d(in_channels = planes//4,out_channels = planes//4,kernel_size = convsize+2,stride = convstr,padding = convpadding+1),
            nn.BatchNorm1d(planes//4))
        self.conv1_3 = nn.Sequential(
            nn.Conv1d(inplanes, planes//4, kernel_size=1, stride = convstr, padding=0, bias=False),
            nn.BatchNorm1d(planes//4),
            nn.LeakyReLU(0.2,),
            nn.Conv1d(in_channels = planes//4,out_channels = planes//4,kernel_size = convsize+4,stride = convstr,padding = convpadding+2),
            nn.BatchNorm1d(planes//4))
        self.conv1_4 = nn.Sequential(
            nn.Conv1d(inplanes, planes//4, kernel_size=1, stride = convstr, padding=0, bias=False),
            nn.BatchNorm1d(planes//4),
            nn.LeakyReLU(0.2,),
            nn.Conv1d(in_channels = planes//4,out_channels = planes//4,kernel_size = convsize+6,stride = convstr,padding = convpadding+3),
            nn.BatchNorm1d(planes//4))
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = self.Inputconv1x1(x)

        c1 = self.conv1_1(x)
        c2 = self.conv1_2(x)
        c3 = self.conv1_3(x)
        c4 = self.conv1_4(x)

        out = torch.cat([c1,c2,c3,c4],1)
        out += residual
        out = self.relu(out)

        return out

class IncUNet (nn.Module):
    def __init__(self, in_shape):
        super(IncUNet, self).__init__()
        in_channels, height, width = in_shape
        self.e1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2,),
            IncResBlock(64,64))
        self.e2 = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv1d(64, 128, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(128),
            IncResBlock(128,128))
        self.e2add = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(128))
        self.e3 = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2,),
            nn.Conv1d(128,256, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(256),
            IncResBlock(256,256))
        self.e4 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv1d(256,256, kernel_size=4 , stride=1 , padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv1d(256,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(512),
            IncResBlock(512,512))
        self.e4add = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv1d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(512)) 
        self.e5 = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv1d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2,),
            nn.Conv1d(512,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(512),
            IncResBlock(512,512))

        self.e6 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv1d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv1d(512,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(512), 
            IncResBlock(512,512))
        
        self.e6add = nn.Sequential(
            nn.Conv1d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(512)) 
        
        self.e7 = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv1d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2,),
            nn.Conv1d(512,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(512),
            IncResBlock(512,512))
        
        self.e8 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv1d(512,512, kernel_size=4, stride=1,padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv1d(512,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(512))
        
        
        self.d1 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(512, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(512, 512, kernel_size=4, stride=1,padding =1),
            nn.BatchNorm1d(512),
            IncResBlock(512,512))
        
        self.d2 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(1024, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(512),
            IncResBlock(512,512))
        
        self.d3 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(1024, 512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            IncResBlock(512,512))
        
        self.d4 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(1024, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(512),
            IncResBlock(512,512))
        
        self.d5 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(1024, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(512),
            IncResBlock(512,512))
        
        self.d6 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(1024, 512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(512),
            IncResBlock(512,512))
        
        self.d7 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(1024, 256, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(256, 256, kernel_size=4, stride=1,padding=1),
            nn.BatchNorm1d(256),
            IncResBlock(256,256))
        
        self.d8 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(512, 128, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(128, 128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(128))
        
        self.d9 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(128))
        
        self.d10 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(256, 64, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(64))
        
        self.out_l = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(128, in_channels, kernel_size=4, stride=2,padding=1))
    
    
    def forward(self, x):
        
        en1 = self.e1(x)
        en2 = self.e2(en1)
        en2add = self.e2add(en2)
        en3 = self.e3(en2add)
        en4 = self.e4(en3)
        en4add = self.e4add(en4)
        en5 = self.e5(en4add)
        en6 = self.e6(en5)
        en6add = self.e6add(en6)
        en7 = self.e7(en6add)
        en8 = self.e8(en7)
        
        de1_ = self.d1(en8)
        de1 = torch.cat([en7,de1_],1)
        de2_ = self.d2(de1)
        de2 = torch.cat([en6add,de2_],1)
        de3_ = self.d3(de2)
        de3 = torch.cat([en6,de3_],1)
        de4_ = self.d4(de3)
        de4 = torch.cat([en5,de4_],1)
        de5_ = self.d5(de4)
        de5 = torch.cat([en4add,de5_],1)
        de6_ = self.d6(de5)
        de6 = torch.cat([en4,de6_],1)
        de7_ = self.d7(de6)
        de7 = torch.cat([en3,de7_],1)
        de8 = self.d8(de7)
        de8_ = self.d8(de7)
        de8 = torch.cat([en2add,de8_],1)
        de9_ = self.d9(de8)
        de9 = torch.cat([en2,de9_],1)
        de10_ = self.d10(de9)
        de10 = torch.cat([en1,de10_],1)
        out = self.out_l(de10)
        
        return out
