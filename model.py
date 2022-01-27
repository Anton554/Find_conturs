import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

from matplotlib import pyplot as plt
import matplotlib.cm as cm


def get_transform():
    transform = transforms.Compose([transforms.Resize((28, 28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                    ])
    return transform


class DtLoader:
    """
    Класс объекта загрузчика
    """
    def __init__(self, train: str, val: str, test: str):
        train_data = torchvision.datasets.ImageFolder(root=train, transform=get_transform())
        val_data = torchvision.datasets.ImageFolder(root=val, transform=get_transform())
        test_data = torchvision.datasets.ImageFolder(root=test, transform=get_transform())
        batch_size = 64
        self.train_data_loader = data.DataLoader(train_data, batch_size, shuffle=True)
        self.val_data_loader = data.DataLoader(val_data, batch_size, shuffle=True)
        self.test_data_loader = data.DataLoader(test_data, 1)


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # создаем 3 полносвязных слоя и
        # указываем колличество входных и выходных нейронов
        # self.fc1 = nn.Linear(28 * 28 * 3, 84)
        self.fc1 = nn.Linear(28 * 28, 84)
        self.fc2 = nn.Linear(84, 50)
        self.fc3 = nn.Linear(50, 10)

    def _img_show(self, x):
        """ Просмотр картинки
        :param x: Tensor
        :return:
        """
        # меняем местами каналы
        y = x.permute(0, 2, 3, 1)
        for i in range(len(y)):
            # [64,28,28,1] -> [1,28,28,1]
            y0 = y.numpy()[i:i + 1, :, :, :]
            # [1,28,28,1] -> [28,28,1]
            y0 = y0.reshape(28, 28, 1)
            y0 = 255 - y0
            plt.imshow(y0, cmap=cm.gray)
            plt.show()

    def forward(self, x):
        # оставляем один канал цвета
        x = x[:, :1, :, :]
        self._img_show(x)
        x = x.reshape(-1, 28 * 28)
        # указываем функцию активации на каждом слое
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
