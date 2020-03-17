def conv_block(in_channels, out_channels, dropout=0, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, *args, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Dropout(dropout)
        )
    

class QuizConNet(nn.Module):
    def __init__(self):
        super(QuizConNet, self).__init__()

        x1_out = 32
        self.x2 = conv_block(in_channels=3, out_channels=x1_out, kernel_size=3, padding=1, bias=False)  
        self.x3 = conv_block(in_channels=35, out_channels=x1_out, kernel_size=3, padding=1, dropout=0.1,  bias=False)     
        # self.x3 = conv_block(in_channels=x1_out, out_channels=x1_out, kernel_size=3, padding=1, dropout=0.1,  bias=False)     
        self.x4 = nn.MaxPool2d(2, 2)

        x5_out = 128
        self.x5 = conv_block(in_channels=67, out_channels=x5_out, kernel_size=3, padding=1, bias=False)  
        self.x6 = conv_block(in_channels=67+x5_out, out_channels=x5_out, kernel_size=3, padding=1, dropout=0.1,  bias=False)     
        self.x7 = conv_block(in_channels=67+x5_out+x5_out, out_channels=x5_out, kernel_size=3, padding=1, dropout=0.1,  bias=False)     
        self.x8 = nn.MaxPool2d(2, 2)


        x9_out = 256
        self.x9 = conv_block(in_channels=384, out_channels=x9_out, kernel_size=3, padding=1, bias=False)  
        self.x10 = conv_block(in_channels=384+x9_out, out_channels=x9_out, kernel_size=3, padding=1, dropout=0.1,  bias=False)     
        self.x11 = conv_block(in_channels=384+x9_out+x9_out, out_channels=x9_out, kernel_size=3, padding=1, dropout=0.1,  bias=False)     

        self.x12 = nn.AvgPool2d(kernel_size=8)                                                                          
        self.x13 = nn.Conv2d(in_channels=x9_out, out_channels=10, kernel_size=1, bias=False)                              

    def forward(self, x1):
        
        # x1 = Input
        x2 = self.x2(x1)  # inp : 32x32x3 -> 32x32x32
        x3 = self.x3(torch.cat((x1, x2), dim=1))  # 32x32x(3+32) -> 32x32x32
        x4 = self.x4(torch.cat((x1, x2, x3), dim=1))  # 32x32x(3+32+32) -> 16x16x(3+32+32)

        x5 = self.x5(x4)  # 16x16x67 -> 16x16x128
        x6 = self.x6(torch.cat((x4, x5), dim=1)) # 16x16x(67+128) -> 16x16x128
        x7 = self.x7(torch.cat((x4, x5, x6), dim=1))  # 16x16x(67+128+128) -> 16x16x128
        x8 = self.x8(torch.cat((x5, x6, x7), dim=1))  # 16x16x(128*3) -> 8x8x384(128*3) 

        x9  = self.x9(x8)  # 8x8x384 -> 8x8x256
        x10 = self.x10(torch.cat((x8, x9), dim=1))  # 8x8x(384+256) -> 8x8x256
        x11 = self.x11(torch.cat((x8, x9, x10), dim=1))  # 8x8x(384+256+256) -> 8x8x256
        x12 = self.x12(x11)  # 8x8x256 -> 1x1x256
        x13 = self.x13(x12)  # 1x1x256 -> 1x1x10
        x = x13.view(-1, 10)
        return F.log_softmax(x)



class QuizAddNet(nn.Module):
    def __init__(self):
        super(QuizAddNet, self).__init__()

        x1_out = 3#32
        # self.x1 = conv_block(in_channels=3, out_channels=x1_out, kernel_size=3, padding=1, bias=False)  
        self.x2 = conv_block(in_channels=3, out_channels=x1_out, kernel_size=3, padding=1, bias=False)  
        self.x3 = conv_block(in_channels=x1_out, out_channels=x1_out, kernel_size=3, padding=1, dropout=0.1,  bias=False)     
        self.x4 = nn.MaxPool2d(2, 2)

        x5_out = 3 #64
        self.x5 = conv_block(in_channels=x1_out, out_channels=x5_out, kernel_size=3, padding=1, bias=False)  
        self.x6 = conv_block(in_channels=x5_out, out_channels=x5_out, kernel_size=3, padding=1, dropout=0.1,  bias=False)     
        self.x7 = conv_block(in_channels=x5_out, out_channels=x5_out, kernel_size=3, padding=1, dropout=0.1,  bias=False)     
        self.x8 = nn.MaxPool2d(2, 2)


        x9_out = 3#128
        x11_out = 3#256
        self.x9 = conv_block(in_channels=x5_out, out_channels=x9_out, kernel_size=3, padding=1, bias=False)  
        self.x10 = conv_block(in_channels=x9_out, out_channels=x9_out, kernel_size=3, padding=1, dropout=0.1,  bias=False)     
        self.x11 = conv_block(in_channels=x9_out, out_channels=x9_out, kernel_size=3, padding=1, dropout=0.1,  bias=False)     

        self.x12 = nn.AvgPool2d(kernel_size=8)                                                                          
        self.x13 = nn.Conv2d(in_channels=x11_out, out_channels=10, kernel_size=1, bias=False)                              

    def forward(self, x1):
        # x1 = Input
        x2 = self.x2(x1)
        x3 = self.x3(x1+x2) 
        x4 = self.x4(x1+x2+x3)

        x5 = self.x5(x4)
        x6 = self.x6(x4+x5)
        x7 = self.x7(x4+x5+x6) 
        x8 = self.x8(x5+x6+x7)

        x9  = self.x9(x8)
        x10 = self.x10(x8+x9)
        x11 = self.x11(x8+x9+x10) 
        x12 = self.x12(x11)
        x13 = self.x13(x12)
        x = x13.view(-1, 10)
        return F.log_softmax(x)

        
