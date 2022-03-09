import torch
import model_cnn
import cv2 as cv


def print_proc(proc):
    tx = ''
    for num, el in enumerate(proc, start=0):
        tx += f'{num} - {round(el * 100, 4)} %\n'
    return tx


def print_proc_fin(proc):
    return round(max(proc) * 100, 3)


def predict(net, img):
    """ Классификация изобравения при помощи модели CNN

    :param img: np_arr или путь к картинке png
    :return: номер предсказанного класса и список процентов вероатности отнесения к
            конкретному классу
    """
    if isinstance(img, str):
        np_arr = cv.imread(img) # shape [28, 28, 3]
    else:
        np_arr = img
    transform = model_cnn.get_transform()
    tr = transform(np_arr)
    input = tr.reshape(1, 3, 28, 28)
    pred = net(input)
    pred = torch.softmax(pred, dim=1)
    proc = [tn.item() for tn in pred[0]]
    pred = pred.argmax()
    return pred.item(), proc




if __name__ == '__main__':
    net = torch.load('C:/Projects/IT/Python/Find_conturs/net_cnn/cnn_net6_7_9_97.pth')
    pred, proc = predict(net, './img/fin/8-img_51.png')
    # pred, proc = predict(net_cnn, './img/predict/num-362.jpg')
    print(print_proc(proc))
    print(f'Ваше число - {pred}')
