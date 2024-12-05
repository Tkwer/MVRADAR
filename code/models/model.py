import torch
import torch.nn as nn
from torch.autograd import Variable
import math
from models.functions import ReverseLayerF

# 这部分是作为特征增加的实验验证 5 个
# RT 
# DT
# RDT
# ART
# RT + DT + ART
class ContrastModel(nn.Module):
    def __init__(self, num_classes, modelName, lstm_layers, hidden_size, fc_size):
        super(ContrastModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc_size = fc_size
        self.modelName = modelName
        if 'RT+DT+ART_MFFNet' in modelName:
            inputsize = num_classes*3		
        else:
            inputsize = num_classes		

        self.feature_2D = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(num_features=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(3, 6, kernel_size=3, padding=1),
            # nn.BatchNorm2d(num_features=6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # self.fc_pre1 到 self.fc_pre3 都是为了对齐输出
        self.fc_pre1 = nn.Sequential(
            nn.Linear(1980, fc_size),#4480,依据卷积后的大小修改
            nn.Dropout()
        )
        self.fc_pre3 = nn.Sequential(
            nn.Linear(1980, fc_size),#4480,依据卷积后的大小修改
            nn.Dropout()
        )
        self.fc_pre4 = nn.Sequential(
            nn.Linear(1350, fc_size),#4480,依据卷积后的大小修改
            nn.Dropout()
        )
        
        self.rnn = nn.LSTM(input_size = fc_size,#256 * 6 * 6,
                    hidden_size = hidden_size,
                    num_layers = lstm_layers,
                    bidirectional = True,
                    batch_first = True)

        self.fc134 = nn.Sequential(
            nn.Linear(hidden_size*2, self.num_classes),
            nn.Dropout()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(180, self.num_classes),
            nn.Dropout()
        )			
        self.fc5 = nn.Sequential(
            nn.Linear(990, self.num_classes),
            nn.Dropout()
        )	
        self.fc_out = nn.Sequential(
            nn.Linear(inputsize, self.num_classes),
            # nn.Dropout(0.3)
        )	

        
    def forward(self, input_var1,input_var2,input_var3,input_var4,input_var5, hidden=None, steps=0):
        '''
        inputs: [ART_feature, DT_feature, ERT_feature, RDT_feature, RT_feature]
        '''

        length = len(input_var1)
        fs = Variable(torch.zeros(length, 3, input_var1[0].size(0), self.rnn.input_size)).cuda()
        for i in range(length):
            # print self.features(inputs[i].unsqueeze(0)).shape
            f1 = self.feature_2D(input_var1[i])
            f3 = self.feature_2D(input_var3[i])
            f4 = self.feature_2D(input_var4[i])
            # print(f.size(0), -1)
            # print(input_var1[i].size()) # (12,1,91,40)
            f1 = f1.view(f1.size(0), -1)
            # print(f1.size()) # (12,1188)
            f1 = self.fc_pre1(f1)
            f3 = f3.view(f3.size(0), -1)
            f3 = self.fc_pre3(f3)
            f4 = f4.view(f4.size(0), -1)
            f4 = self.fc_pre4(f4)
            fs[i, 0, :, :] = f1
            fs[i, 1, :, :] = f3
            fs[i, 2, :, :] = f4
        output1, hidden = self.rnn(fs[:,0,...], hidden)
        output3, hidden = self.rnn(fs[:,1,...], hidden)
        output4, hidden = self.rnn(fs[:,2,...], hidden)

        f2 = self.feature_2D(input_var2)
        f2 = f2.view(f2.size(0), -1)
        f5 = self.feature_2D(input_var5)
        f5 = f5.view(f5.size(0), -1)
        # f2 = self.dropout(f2) 
        # f5 = self.dropout(f5)
        output2	= self.fc2(f2)
        output5	= self.fc5(f5)
        output1 = self.fc134(output1[:, -1, :])
        output3 = self.fc134(output3[:, -1, :])
        output4 = self.fc134(output4[:, -1, :])
        
        if 'RT_CNN' in self.modelName and 'RT_CNN-LSTM' not in self.modelName:
            outputs = self.fc_out(output5)
            return outputs

        elif 'DT_CNN' in self.modelName and 'RDT_CNN-LSTM' not in self.modelName:
            outputs = self.fc_out(output2)
            return outputs

        elif 'RDT_CNN-LSTM' in self.modelName:	
            outputs = self.fc_out(output4)
            return outputs

        elif 'ART_CNN-LSTM' in self.modelName:	
            outputs = self.fc_out(output1)
            return outputs

        elif 'ERT_CNN-LSTM' in self.modelName:	
            outputs = self.fc_out(output3)
            return outputs

        elif 'RT+DT+ART_MFFNet' in self.modelName:	
            outputs = torch.cat((output1,output2,output5),1)
            # print(outputs.size())
            outputs = self.fc_out(outputs)
            return outputs



# --------2022.5.28-----
#      新增的网络
class EfficientChannelAttention(nn.Module):           # Efficient Channel Attention module
    def __init__(self, c, b=1, gamma=2):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x.transpose(-1, -2)).transpose(-1, -2)
        weights = self.sigmoid(x)

        return weights


# nn.Sigmoid(), 改为 nn.Softmax(),
class SENet(nn.Module):
    def __init__(self, c, r=7):
        super(SENet, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.se = nn.Sequential(
            nn.Linear(c, int(c / r)),  # int(channel * r)取整数
            nn.ReLU(),
            nn.Linear(int(c / r), 5),
            nn.Softmax(),
        )
        
    def forward(self, x):
        # weight = torch.var(x, dim=-1, unbiased=False)
        # weight = self.avg_pool(x)
        # weight = weight.view(weight.size(0), -1)
        weight = x.view(x.size(0), -1)
        weight = self.se(weight)
        weights = torch.unsqueeze(weight, dim=-1)
        # h, w = weight.shape
        # weights = torch.reshape(weight, (h, w, 1))
        return weights

# orignal-MFFNet （multiple feature fusion）
# SE-MFFNet
# ECA-MFFNet
# 加入 loss 训练控制senet的权重的 KA-SE-MFFNet （Knowledge Aid）
# 加入 loss 训练控制senet的权重的 KA-ECA-MFFNet

class MFFNet(nn.Module):

    def __init__(self, num_classes, modelName, lstm_layers, hidden_size, fc_size):
        super(MFFNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc_size = fc_size
        self.modelName = modelName

        self.feature_2D = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(num_features=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(3, 6, kernel_size=3, padding=1),
            # nn.BatchNorm2d(num_features=6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(6, 1, kernel_size=3, padding=1),
            # nn.BatchNorm2d(num_features=6),
            nn.ReLU(inplace=True),
        )
        # self.fc_pre1 到 self.fc_pre3 都是为了对齐输出
        self.fc_pre1 = nn.Sequential(
            nn.Linear(198, fc_size),#4480,依据卷积后的大小修改
            nn.Dropout()
        )
        self.fc_pre3 = nn.Sequential(
            nn.Linear(198, fc_size),#4480,依据卷积后的大小修改
            nn.Dropout()
        )
        self.fc_pre4 = nn.Sequential(
            nn.Linear(225, fc_size),#4480,依据卷积后的大小修改
            nn.Dropout()
        )
        
        self.rnn = nn.LSTM(input_size = fc_size,#256 * 6 * 6,
                    hidden_size = hidden_size,
                    num_layers = lstm_layers,
                    bidirectional = True,
                    batch_first = True)
        
        # ECA 和 SE 模块
        if 'ECA' in modelName:
            self.channel = EfficientChannelAttention(5)
        elif 'SE' in modelName:
            self.channel = SENet(5*7)
        
        # self.Softmax = nn.Softmax(dim=1)

        self.fc134 = nn.Sequential(
            nn.Linear(hidden_size*2, self.num_classes),
            nn.Dropout()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(30, self.num_classes),
            nn.Dropout()
        )			
        self.fc5 = nn.Sequential(
            nn.Linear(165, self.num_classes),
            nn.Dropout()
        )	
        self.fc_out = nn.Sequential(
            nn.Linear(35, self.num_classes),
            # 2022-10-14tianjia
            nn.Dropout(0.2)
        )	

        
    def forward(self, input_var1,input_var2,input_var3,input_var4,input_var5, hidden=None, steps=0):
        '''
        inputs: [ART_feature, DT_feature, ERT_feature, RDT_feature, RT_feature]
        '''

        length = len(input_var1)
        fs = Variable(torch.zeros(length, 3, input_var1[0].size(0), self.rnn.input_size)).cuda()
        for i in range(length):
            # print self.features(inputs[i].unsqueeze(0)).shape
            f1 = self.feature_2D(input_var1[i])
            f3 = self.feature_2D(input_var3[i])
            f4 = self.feature_2D(input_var4[i])
            # print(f.size(0), -1)
            f1 = f1.view(f1.size(0), -1)
            f3 = f3.view(f3.size(0), -1)
            f4 = f4.view(f4.size(0), -1)
            f1 = self.fc_pre1(f1)
            f3 = self.fc_pre3(f3)
            f4 = self.fc_pre4(f4)
            fs[i, 0, :, :] = f1
            fs[i, 1, :, :] = f3
            fs[i, 2, :, :] = f4
        output1, hidden = self.rnn(fs[:,0,...], hidden)
        output3, hidden = self.rnn(fs[:,1,...], hidden)
        output4, hidden = self.rnn(fs[:,2,...], hidden)

        f2 = self.feature_2D(input_var2)
        f2 = f2.view(f2.size(0), -1)
        f5 = self.feature_2D(input_var5)
        f5 = f5.view(f5.size(0), -1)
        # f2 = self.dropout(f2) 
        # f5 = self.dropout(f5)
        output2	= self.fc2(f2)
        output5	= self.fc5(f5)
        output1 = self.fc134(output1[:, -1, :])
        output3 = self.fc134(output3[:, -1, :])
        output4 = self.fc134(output4[:, -1, :])

        if 'ALL_MFFNet' in self.modelName:
            outputs = torch.cat((output1,output3,output4,output2,output5),1)
            outputs = self.fc_out(outputs)
            return outputs
        else:
            outputs = torch.stack((output1,output3,output4,output2,output5),dim = 1)
            weights = self.channel(outputs)
            # weights = self.Softmax(weights)
            outputs = outputs * weights
            outputs = outputs.view(outputs.size(0), -1)
            outputs = self.fc_out(outputs)
            if 'KA' in self.modelName:
                return outputs,weights.squeeze(-1),output1,output3,output4,output2,output5
            else:
                return outputs
                
class FuzzyModel(nn.Module):
    def __init__(self, num_classes, modelName, lstm_layers, hidden_size, fc_size):
        super(FuzzyModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc_size = fc_size
        self.modelName = modelName
        inputsize2 = num_classes*2	
        inputsize = num_classes		

        self.feature_2D = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(num_features=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(3, 6, kernel_size=3, padding=1),
            # nn.BatchNorm2d(num_features=6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # self.fc_pre1 到 self.fc_pre3 都是为了对齐输出
        self.fc_pre1 = nn.Sequential(
            nn.Linear(1188, fc_size),#4480,依据卷积后的大小修改
            nn.Dropout()
        )
        self.fc_pre3 = nn.Sequential(
            nn.Linear(1188, fc_size),#4480,依据卷积后的大小修改
            nn.Dropout()
        )
        self.fc_pre4 = nn.Sequential(
            nn.Linear(1350, fc_size),#4480,依据卷积后的大小修改
            nn.Dropout()
        )
        
        self.rnn = nn.LSTM(input_size = fc_size,#256 * 6 * 6,
                    hidden_size = hidden_size,
                    num_layers = lstm_layers,
                    bidirectional = True,
                    batch_first = True)

        self.fc134 = nn.Sequential(
            nn.Linear(hidden_size*2, self.num_classes),
            nn.Dropout()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(180, self.num_classes),
            nn.Dropout()
        )			
        self.fc5 = nn.Sequential(
            nn.Linear(990, self.num_classes),
            nn.Dropout()
        )	
        self.fc_out1 = nn.Sequential(
            nn.Linear(inputsize*3, 3),
            # nn.Dropout(0.3)
        )	
        self.fc_out2 = nn.Sequential(
            nn.Linear(inputsize2, 3),
            # nn.Dropout(0.3)
        )	
        self.fc_out3 = nn.Sequential(
            nn.Linear(inputsize2, 3),
            # nn.Dropout(0.3)
        )	
        self.fc_out4 = nn.Sequential(
            nn.Linear(inputsize2, 2),
            # nn.Dropout(0.3)
        )	
        self.fc_out0 = nn.Sequential(
            nn.Linear(inputsize, self.num_classes),
        )    # nn.Dropout(0.3)
        
    def forward(self, input_var1,input_var2,input_var3,input_var4,input_var5, hidden=None, steps=0):
        '''
        inputs: [ART_feature, DT_feature, ERT_feature, RDT_feature, RT_feature]
        '''

        length = len(input_var1)
        fs = Variable(torch.zeros(length, 3, input_var1[0].size(0), self.rnn.input_size)).cuda()
        for i in range(length):
            # print self.features(inputs[i].unsqueeze(0)).shape
            f1 = self.feature_2D(input_var1[i])
            f3 = self.feature_2D(input_var3[i])
            f4 = self.feature_2D(input_var4[i])
            # print(f.size(0), -1)
            # print(input_var1[i].size()) # (12,1,91,40)
            f1 = f1.view(f1.size(0), -1)
            # print(f1.size()) # (12,1188)
            f1 = self.fc_pre1(f1)
            f3 = f3.view(f3.size(0), -1)
            f3 = self.fc_pre3(f3)
            f4 = f4.view(f4.size(0), -1)
            f4 = self.fc_pre4(f4)
            fs[i, 0, :, :] = f1
            fs[i, 1, :, :] = f3
            fs[i, 2, :, :] = f4
        output1, hidden = self.rnn(fs[:,0,...], hidden)
        output3, hidden = self.rnn(fs[:,1,...], hidden)
        output4, hidden = self.rnn(fs[:,2,...], hidden)

        f2 = self.feature_2D(input_var2)
        f2 = f2.view(f2.size(0), -1)
        f5 = self.feature_2D(input_var5)
        f5 = f5.view(f5.size(0), -1)
        # f2 = self.dropout(f2) 
        # f5 = self.dropout(f5)
        output2	= self.fc2(f2)
        output5	= self.fc5(f5)
        output1 = self.fc134(output1[:, -1, :])
        output3 = self.fc134(output3[:, -1, :])
        output4 = self.fc134(output4[:, -1, :])
        
        # Double use DT
        outputs4 = torch.cat((output4,output2),1)
        outputs4 = self.fc_out4(outputs4)
        # outputs4 = self.fc_out4(output4)

        # Up & Down use ERT+RDT
        outputs3 = torch.cat((output4,output3),1)
        outputs3 = self.fc_out3(outputs3)
        # outputs3 = self.fc_out3(output4)

        # Right & Left use ART
        outputs2 = torch.cat((output4,output1),1)
        outputs2 = self.fc_out2(outputs2)
        # outputs2 = self.fc_out2(output4)

        # Front & Back use RT+DT+ART
        outputs1 = torch.cat((output1,output2,output5),1)
        # outputs1 = torch.cat((output1,output4),1)
        # print(outputs1.size()) # [10,21]
        outputs1 = self.fc_out1(outputs1)
        # outputs1 = self.fc_out1(output4)

        return outputs1,outputs2,outputs3,outputs4
        
class FuzzyModel_V2(nn.Module):

    def __init__(self, num_classes, modelName, lstm_layers, hidden_size, fc_size,numsourcedomain):
        super(FuzzyModel_V2, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc_size = fc_size
        self.modelName = modelName
        inputsize2 = num_classes*2	
        inputsize = num_classes
        self.numsourcedomain = numsourcedomain
        # print(num_classes)
        self.feature_2D = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(num_features=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(num_features=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(num_features=6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # nn.Conv2d(6, 3, kernel_size=3, padding=1),
            # # nn.BatchNorm2d(num_features=6),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            # nn.BatchNorm2d(num_features=1),
            nn.ReLU(inplace=True),
        )
        # self.fc_pre1 到 self.fc_pre3 都是为了对齐输出
        self.fc_pre1 = nn.Sequential(
            nn.Linear(198, fc_size),#198,依据卷积后的大小修改140
            nn.Dropout()
        )
        self.fc_pre3 = nn.Sequential(
            nn.Linear(198, fc_size),#198,依据卷积后的大小修改140
            nn.Dropout()
        )
        self.fc_pre4 = nn.Sequential(
            nn.Linear(225, fc_size),#225,依据卷积后的大小修改169
            nn.Dropout()
        )
        self.fc_pre4_2 = nn.Sequential(
            nn.Linear(1350, fc_size),#225,依据卷积后的大小修改169
            nn.Dropout()
        )
        
        self.rnn = nn.LSTM(input_size = fc_size,#256 * 6 * 6,
                    hidden_size = hidden_size,
                    num_layers = lstm_layers,
                    bidirectional = True,
                    batch_first = True)
        # bidirectional = True
        
        # self.Softmax = nn.Softmax(dim=1)

        self.fc134 = nn.Sequential(
            nn.Linear(hidden_size*2, self.num_classes),
            nn.Dropout()
        )
        # hidden_size*2
        self.fc2 = nn.Sequential(
            nn.Linear(30, self.num_classes),
            nn.Dropout()
        )			
        self.fc5 = nn.Sequential(
            nn.Linear(165, self.num_classes),
            nn.Dropout()
        )	
        self.fc_out1 = nn.Sequential(
            # nn.Linear(inputsize*3, int(inputsize*1.5)),
            # nn.Linear(int(inputsize*1.5), 3),
            nn.Linear(inputsize*3, 3),
            nn.Dropout(0.2),
            # nn.ReLU(),
            # nn.Softmax(),
        )	
        self.fc_out2 = nn.Sequential(
            # nn.Linear(inputsize2, inputsize),
            # nn.Linear(inputsize, 3),
            nn.Linear(inputsize2, 3),
            nn.Dropout(0.2),
            # nn.ReLU(),
            # nn.Softmax(),
        )	
        self.fc_out3 = nn.Sequential(
            # nn.Linear(inputsize2, inputsize),
            # nn.Linear(inputsize, 3),
            nn.Linear(inputsize*3, 3),
            nn.Dropout(0.2),
            # nn.ReLU(),
            # nn.Softmax(),
        )	
        self.fc_out4 = nn.Sequential(
            # nn.Linear(inputsize2, inputsize),
            # nn.Linear(inputsize, 2),
            nn.Linear(inputsize2, 2),
            nn.Dropout(0.2),
            # nn.ReLU(),
            # nn.Softmax(),
        )	
        self.feature_2D_2 = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(num_features=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(3, 6, kernel_size=3, padding=1),
            # nn.BatchNorm2d(num_features=6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc2_2 = nn.Sequential(
            nn.Linear(180, self.num_classes),
            nn.Dropout()
        )	
        self.domain_classifier = nn.Sequential(

        )
        self.domain_classifier.add_module('d_fc1', nn.Linear(11, 6))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(6))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(6, 5))# TODO: Num 5 people sample
        self.domain_classifier.add_module('d_dropout', nn.Dropout(0.2))
        # self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))
        
    def forward(self, input_var1,input_var2,input_var3,input_var4,input_var5, hidden=None, steps=0):
        '''
        inputs: [ART_feature, DT_feature, ERT_feature, RDT_feature, RT_feature]
        '''

        length = len(input_var1)
        fs = Variable(torch.zeros(length, 3, input_var1[0].size(0), self.rnn.input_size)).cuda()
        fs_2 = Variable(torch.zeros(length, 3, input_var1[0].size(0), self.rnn.input_size)).cuda()
        for i in range(length):
            # print self.features(inputs[i].unsqueeze(0)).shape
            f1 = self.feature_2D(input_var1[i])
            f3 = self.feature_2D(input_var3[i])
            f4 = self.feature_2D(input_var4[i])

            # print(f.size(0), -1)
            f1 = f1.view(f1.size(0), -1)
            f3 = f3.view(f3.size(0), -1)
            f4 = f4.view(f4.size(0), -1)
            # print(f1.size())
            # print(f3.size())
            # print(f4.size())
            f1 = self.fc_pre1(f1)
            f3 = self.fc_pre3(f3)
            f4 = self.fc_pre4(f4)
            fs[i, 0, :, :] = f1
            fs[i, 1, :, :] = f3
            fs[i, 2, :, :] = f4

            # for Doubleclick
            f4_2 = self.feature_2D_2(input_var4[i])
            f4_2 = f4_2.view(f4_2.size(0), -1)
            # print(f4_2.size())
            f4_2 = self.fc_pre4_2(f4_2)
            fs_2[i, 2, :, :] = f4_2
            # End

        output1, hidden = self.rnn(fs[:,0,...], hidden)
        output3, hidden = self.rnn(fs[:,1,...], hidden)
        output4, hidden = self.rnn(fs[:,2,...], hidden)


        f2 = self.feature_2D(input_var2)
        f2 = f2.view(f2.size(0), -1)
        f5 = self.feature_2D(input_var5)
        f5 = f5.view(f5.size(0), -1)
        # f2 = self.dropout(f2) 
        # f5 = self.dropout(f5)
        output2	= self.fc2(f2)
        output5	= self.fc5(f5)
        output1 = self.fc134(output1[:, -1, :])
        output3 = self.fc134(output3[:, -1, :])
        output4 = self.fc134(output4[:, -1, :])

        # for Doubleclick
        output4_2, hidden = self.rnn(fs_2[:,2,...], hidden)
        output4_2 = self.fc134(output4_2[:, -1, :])
        f2_2 = self.feature_2D_2(input_var2)
        f2_2 = f2_2.view(f2_2.size(0), -1)
        output2_2	= self.fc2_2(f2_2)
        # End
        
        # Double use DT
        outputs4 = torch.cat((output4_2,output2_2),1)
        outputs4 = self.fc_out4(outputs4)
        # outputs4 = self.fc_out4(output4)

        # Up & Down use ERT+RDT
        outputs3 = torch.cat((output4,output3,output1),1)
        outputs3 = self.fc_out3(outputs3)
        # outputs3 = self.fc_out3(output4)

        # Right & Left use ART
        outputs2 = torch.cat((output4,output1),1)
        outputs2 = self.fc_out2(outputs2)
        # outputs2 = self.fc_out2(output4)

        # Front & Back use RT+DT+ART
        # outputs1 = torch.cat((output1,output2,output5),1)
        outputs1 = torch.cat((output1,output2,output5),1)
        # outputs1 = torch.cat((output1,output4),1)
        # print(outputs1.size()) # [10,21]
        outputs1 = self.fc_out1(outputs1)
        # outputs1 = self.fc_out1(output4)
        feature = torch.cat((outputs1,outputs2,outputs3,outputs4),1)

        reverse_feature = ReverseLayerF.apply(feature, 1)
        domain_output = self.domain_classifier(reverse_feature)
        return outputs1,outputs2,outputs3,outputs4, domain_output

# 备份完整模型 #
# class FuzzyModel_V2(nn.Module):

#     def __init__(self, num_classes, modelName, lstm_layers, hidden_size, fc_size):
#         super(FuzzyModel_V2, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_classes = num_classes
#         self.fc_size = fc_size
#         self.modelName = modelName
#         inputsize2 = num_classes*2	
#         inputsize = num_classes
#         print(num_classes)
#         self.feature_2D = nn.Sequential(
#             nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
#             # nn.BatchNorm2d(num_features=3),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(3, 6, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(num_features=6),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(6, 9, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(num_features=6),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             # nn.Conv2d(6, 3, kernel_size=3, padding=1),
#             # # nn.BatchNorm2d(num_features=6),
#             # nn.ReLU(inplace=True),
#             # nn.MaxPool2d(kernel_size=3, stride=1),
#             nn.Conv2d(9, 1, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(num_features=1),
#             nn.ReLU(inplace=True),
#         )
#         # self.fc_pre1 到 self.fc_pre3 都是为了对齐输出
#         self.fc_pre1 = nn.Sequential(
#             nn.Linear(198, fc_size),#198,依据卷积后的大小修改140
#             nn.Dropout()
#         )
#         self.fc_pre3 = nn.Sequential(
#             nn.Linear(198, fc_size),#198,依据卷积后的大小修改140
#             nn.Dropout()
#         )
#         self.fc_pre4 = nn.Sequential(
#             nn.Linear(225, fc_size),#225,依据卷积后的大小修改169
#             nn.Dropout()
#         )
#         self.fc_pre4_2 = nn.Sequential(
#             nn.Linear(1350, fc_size),#225,依据卷积后的大小修改169
#             nn.Dropout()
#         )
        
#         self.rnn = nn.LSTM(input_size = fc_size,#256 * 6 * 6,
#                     hidden_size = hidden_size,
#                     num_layers = lstm_layers,
#                     bidirectional = True,
#                     batch_first = True)
#         # bidirectional = True
        
#         # self.Softmax = nn.Softmax(dim=1)

#         self.fc134 = nn.Sequential(
#             nn.Linear(hidden_size*2, self.num_classes),
#             nn.Dropout()
#         )
#         # hidden_size*2
#         self.fc2 = nn.Sequential(
#             nn.Linear(30, self.num_classes),
#             nn.Dropout()
#         )			
#         self.fc5 = nn.Sequential(
#             nn.Linear(165, self.num_classes),
#             nn.Dropout()
#         )	
#         self.fc_out1 = nn.Sequential(
#             # nn.Linear(inputsize*3, int(inputsize*1.5)),
#             # nn.Linear(int(inputsize*1.5), 3),
#             nn.Linear(inputsize*3, 3),
#             # nn.Dropout(0.3)
#         )	
#         self.fc_out2 = nn.Sequential(
#             # nn.Linear(inputsize2, inputsize),
#             # nn.Linear(inputsize, 3),
#             nn.Linear(inputsize2, 3),
#             # nn.Dropout(0.3)
#         )	
#         self.fc_out3 = nn.Sequential(
#             # nn.Linear(inputsize2, inputsize),
#             # nn.Linear(inputsize, 3),
#             nn.Linear(inputsize2, 3),
#             # nn.Dropout(0.3)
#         )	
#         self.fc_out4 = nn.Sequential(
#             # nn.Linear(inputsize2, inputsize),
#             # nn.Linear(inputsize, 2),
#             nn.Linear(inputsize2, 2),
#             # nn.Dropout(0.3)
#         )	
#         self.feature_2D_2 = nn.Sequential(
#             nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
#             # nn.BatchNorm2d(num_features=3),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(3, 6, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(num_features=6),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         self.fc2_2 = nn.Sequential(
#             nn.Linear(180, self.num_classes),
#             nn.Dropout()
#         )	
        
#     def forward(self, input_var1,input_var2,input_var3,input_var4,input_var5, hidden=None, steps=0):
#         '''
#         inputs: [ART_feature, DT_feature, ERT_feature, RDT_feature, RT_feature]
#         '''

#         length = len(input_var1)
#         fs = Variable(torch.zeros(length, 3, input_var1[0].size(0), self.rnn.input_size)).cuda()
#         fs_2 = Variable(torch.zeros(length, 3, input_var1[0].size(0), self.rnn.input_size)).cuda()
#         for i in range(length):
#             # print self.features(inputs[i].unsqueeze(0)).shape
#             f1 = self.feature_2D(input_var1[i])
#             f3 = self.feature_2D(input_var3[i])
#             f4 = self.feature_2D(input_var4[i])

#             # print(f.size(0), -1)
#             f1 = f1.view(f1.size(0), -1)
#             f3 = f3.view(f3.size(0), -1)
#             f4 = f4.view(f4.size(0), -1)
#             # print(f1.size())
#             # print(f3.size())
#             # print(f4.size())
#             f1 = self.fc_pre1(f1)
#             f3 = self.fc_pre3(f3)
#             f4 = self.fc_pre4(f4)
#             fs[i, 0, :, :] = f1
#             fs[i, 1, :, :] = f3
#             fs[i, 2, :, :] = f4

#             # for Doubleclick
#             f4_2 = self.feature_2D_2(input_var4[i])
#             f4_2 = f4_2.view(f4_2.size(0), -1)
#             # print(f4_2.size())
#             f4_2 = self.fc_pre4_2(f4_2)
#             fs_2[i, 2, :, :] = f4_2
#             # End

#         output1, hidden = self.rnn(fs[:,0,...], hidden)
#         output3, hidden = self.rnn(fs[:,1,...], hidden)
#         output4, hidden = self.rnn(fs[:,2,...], hidden)


#         f2 = self.feature_2D(input_var2)
#         f2 = f2.view(f2.size(0), -1)
#         f5 = self.feature_2D(input_var5)
#         f5 = f5.view(f5.size(0), -1)
#         # f2 = self.dropout(f2) 
#         # f5 = self.dropout(f5)
#         output2	= self.fc2(f2)
#         output5	= self.fc5(f5)
#         output1 = self.fc134(output1[:, -1, :])
#         output3 = self.fc134(output3[:, -1, :])
#         output4 = self.fc134(output4[:, -1, :])

#         # for Doubleclick
#         output4_2, hidden = self.rnn(fs_2[:,2,...], hidden)
#         output4_2 = self.fc134(output4_2[:, -1, :])
#         f2_2 = self.feature_2D_2(input_var2)
#         f2_2 = f2_2.view(f2_2.size(0), -1)
#         output2_2	= self.fc2_2(f2_2)
#         # End
        
#         # Double use DT
#         outputs4 = torch.cat((output4_2,output2_2),1)
#         outputs4 = self.fc_out4(outputs4)
#         # outputs4 = self.fc_out4(output4)

#         # Up & Down use ERT+RDT
#         outputs3 = torch.cat((output4,output3),1)
#         outputs3 = self.fc_out3(outputs3)
#         # outputs3 = self.fc_out3(output4)

#         # Right & Left use ART
#         outputs2 = torch.cat((output4,output1),1)
#         outputs2 = self.fc_out2(outputs2)
#         # outputs2 = self.fc_out2(output4)

#         # Front & Back use RT+DT+ART
#         # outputs1 = torch.cat((output1,output2,output5),1)
#         outputs1 = torch.cat((output1,output2,output5),1)
#         # outputs1 = torch.cat((output1,output4),1)
#         # print(outputs1.size()) # [10,21]
#         outputs1 = self.fc_out1(outputs1)
#         # outputs1 = self.fc_out1(output4)

#         return outputs1,outputs2,outputs3,outputs4
