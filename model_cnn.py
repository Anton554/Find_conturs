import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

from matplotlib import pyplot as plt
import matplotlib.cm as cm



def get_transform():
    # transform = transforms.Compose([transforms.Resize((28, 28)),
    #                             transforms.ToTensor(),
    #                             transforms.Normalize((0.5,), (0.5,))
    #                             ])
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                                ])
    return transform


class DtLoader:
    """
    Класс объекта загрузчика
    """
    # trainset = torchvision.datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
    # valset = torchvision.datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
    # train_data_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    # val_data_loader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=True)

    def __init__(self, train:str, val:str, test:str):
        train_data = torchvision.datasets.ImageFolder(root=train, transform=get_transform())
        val_data = torchvision.datasets.ImageFolder(root=val, transform=get_transform())
        # test_data = torchvision.datasets.DatasetFolder
        test_data = torchvision.datasets.ImageFolder(root=test, transform=get_transform())
        batch_size = 64
        self.train_data_loader = data.DataLoader(train_data, batch_size, shuffle=True)
        self.val_data_loader = data.DataLoader(val_data, batch_size, shuffle=True)
        self.test_data_loader = data.DataLoader(test_data, 1)



class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        # layer1
        # Conv2d - 3 x 28 x 28  -> 32 x 28 x 28
        # MaxPool2d - 32 x 28 x 28  -> 32 x 14 x 14
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        # layer2
        # Conv2d - 32 x 14 x 14  -> 64 x 14 x 14
        # MaxPool2d - 64 x 14 x 14  -> 64 x 7 x 7
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)


    def _img_show(self, x):
        """ Просмотр картинки
        :param x: Tensor
        :return:
        """
        # оставляем один канал цвета
        x = x[:,:1,:,:]
        # меняем местами каналы
        y = x.permute(0, 2, 3, 1)
        for i in range(len(y)):
            # [64,28,28,1] -> [1,28,28,1]
            y0 = y.numpy()[i:i+1, :, :, :]
            # [1,28,28,1] -> [28,28,1]
            y0 = y0.reshape(28, 28, 1)
            y0 = 255 - y0
            plt.imshow(y0, cmap=cm.gray)
            plt.show()


    def forward(self, x):
        # x -> [1, 3, 28, 28]
        # оставляем один канал цвета
        # print(f'{x.shape=}')
        # self._img_show(x)
        # x = x[:, :1, :, :]
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

