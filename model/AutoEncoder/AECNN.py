import torch
import torch.nn as nn
import os


class AECNN(nn.Module):
    def __init__(self, para):
        super(AECNN, self).__init__()
        self.device = para['device']
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.5)
        # encoder
        self.conv1_1 = nn.Conv2d(in_channels=int((1 + para['mutation rate']) * para['user_graph_len']),
                                 out_channels=32 * para['dimension'],
                                 kernel_size=(1, para['dimension']), stride=(1, 1), padding=0)
        self.conv1_3 = nn.Conv2d(in_channels=1, out_channels=8,
                                 kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.conv1_4 = nn.Conv2d(in_channels=8, out_channels=1,
                                 kernel_size=(3, 3), stride=(2, 2), padding=1)


        # decoder
        self.fc1_1 = nn.Linear(in_features=8 * int((para['dimension'] + 3) / 4), out_features=2 * para['dimension'])
        self.fc1_2 = nn.Linear(in_features=2 * para['dimension'], out_features=2 * para['dimension'])
        self.fc1_3 = nn.Linear(in_features=2 * para['dimension'], out_features=2 * para['dimension'])
        self.fc1_4 = nn.Linear(in_features=2 * para['dimension'], out_features=para['dimension'])

    def forward(self, x1):
        # encoder
        x1_1 = self.conv1_1(x1)  # 2048,400,1,1
        x1_1 = x1_1.view(x1_1.size(0), 1, 32, -1)  # 这里x1的size是[batch,16 * para['dimension']]
        x1_2 = self.conv1_3(self.relu(x1_1))
        x1_3 = self.conv1_4(x1_2)
        # decoder
        x1_4 = x1_3.view(x1_3.size(0), -1)
        x1_5 = self.fc1_1(x1_4)
        x1_5 = self.relu(x1_5)
        x1_5 = self.fc1_2(x1_5)
        x1_5 = self.fc1_3(self.relu(x1_5))
        x1_6 = self.fc1_4(x1_5)

        return (x1_1, x1_2, x1_3), (x1_4, x1_5), x1_6


class FusinNet(nn.Module):
    def __init__(self, para):
        super(FusinNet, self).__init__()
        self.dropout = nn.Dropout(0.5)
        # encoder block
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=1,
                                 kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=1, out_channels=8,
                                 kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.conv1_3 = nn.Conv2d(in_channels=16, out_channels=1,
                                 kernel_size=(3, 3), stride=(2, 2), padding=1)
        # decoder block
        self.fc2_1 = nn.Linear(in_features=8 * int((para['dimension'] + 3) / 4), out_features=2 * para['dimension'])
        self.fc2_2 = nn.Linear(in_features=2 * para['dimension'], out_features=para['dimension'])
        self.fc2_3 = nn.Linear(in_features=3 * para['dimension'], out_features=8 * int((para['dimension'] + 3) / 4))
        self.conv2_1 = nn.Conv2d(in_channels=1, out_channels=1,
                                 kernel_size=(3, 3), stride=(1, 1), padding=1)

        # fusion block
        self.relu = nn.ReLU(True)
        self.conv3_1 = nn.Conv2d(in_channels=6, out_channels=1,
                                 kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.fc3_1 = nn.Linear(in_features=8 * int((para['dimension'] + 3) / 4), out_features=para['dimension'])
        # self.fc3_2 = nn.Linear(in_features=para['dimension'], out_features=para['dimension'])
        self.fc3_3 = nn.Linear(in_features=para['dimension'], out_features=1)

    def forward(self, x1, x2, net):
        encoder_feature1, decoder_feature1, _ = net[0](x1)
        encoder_feature2, decoder_feature2, _ = net[1](x2)

        # user_encoder_feature
        x1 = self.conv1_1(encoder_feature1[0])
        x1 = self.conv1_2(x1)
        x1 = torch.cat((x1, encoder_feature1[1]), dim=1)
        x1 = self.relu(x1)
        x1 = self.conv1_3(x1)
        # add encoder graph
        x1 = torch.cat((x1, encoder_feature1[2]), dim=1)

        # service_encoder_feature
        x2 = self.conv1_1(encoder_feature2[0])
        x2 = self.conv1_2(x2)
        x2 = torch.cat((x2, encoder_feature2[1]), dim=1)
        x2 = self.relu(x2)
        x2 = self.conv1_3(x2)
        # add encoder graph
        x2 = torch.cat((x2, encoder_feature2[2]), dim=1)

        # user_decoder_feature
        x3 = self.fc2_1(decoder_feature1[0])
        x3 = self.relu(x3)
        x3 = self.fc2_2(x3)
        x3 = self.dropout(x3)
        x3 = torch.cat((x3, decoder_feature1[1]), dim=1)
        x3 = self.fc2_3(x3)
        x3 = x3.view(x3.size(0), 1, 8, -1)  # 这里x3的size是[batch,16 * para['dimension']]
        x3 = self.conv2_1(x3)

        # services_decoder_feature
        x4 = self.fc2_1(decoder_feature2[0])
        x4 = self.relu(x4)
        x4 = self.fc2_2(x4)
        x4 = self.dropout(x4)
        x4 = torch.cat((x4, decoder_feature2[1]), dim=1)
        x4 = self.fc2_3(x4)
        x4 = x4.view(x4.size(0), 1, 8, -1)  # 这里x3的size是[batch,16 * para['dimension']]
        x4 = self.conv2_1(x4)

        # fusion_feature
        x = torch.cat((x1, x2, x3, x4), dim=1)
        # x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv3_1(x)
        x = x.view(x.size(0), -1)
        x = self.fc3_1(self.relu(x))
        x = self.dropout(x)
        """
        x = self.fc3_2(self.relu(x))
        x = self.dropout(x)
        
        x = self.fc3_2_1(self.relu(x))
        x = self.dropout(x)
        x = self.fc3_2_2(self.relu(x))
        x = self.dropout(x)
        """
        x = self.fc3_3(self.relu(x))

        return x.squeeze()


def save_net(net, epoch=-1):
    if not os.path.isdir('NetParameter'):
        os.makedirs('NetParameter')
    if epoch == -1:
        torch.save(net.state_dict(), 'NetParameter/Net_init')
    else:
        file_path = './NetParameter/Net_' + str(epoch)
        torch.save(net.state_dict(), file_path)
