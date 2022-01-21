import torch
from torch import nn
from torch.nn import functional as F
import model_cnn


class AnNNet(nn.Module):
    def __init__(self):
        super(AnNNet, self).__init__()
        # загрузка обученной CNN
        self.cnn_net_1 = model_cnn.CNNNet()
        self.cnn_net_2 = model_cnn.CNNNet()
        self.cnn_net_1.load_state_dict(torch.load('./net/cnn_net_1.pth'))
        self.cnn_net_2.load_state_dict(torch.load('./net/cnn_net_2.pth'))
        # создаем 3 полносвязных слоя
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 10)

    def pred_net(self, t):
        # Получение общего ветора предсказания от 2-х обученной CNN
        # (32, 3, 28, 28) -> (32, 10) -> (32, 20)
        p_1 = self.cnn_net_1(t)
        p_2 = self.cnn_net_2(t)
        # объединение тензоров по существующим осям
        p_3 = torch.cat((p_1, p_2), axis=-1)
        return p_3

    def forward(self, x):
        # получаем общий ветор предсказания от обученной CNN
        x = self.pred_net(x)
        x = F.leaky_relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
