import torch
import model_cnn
import cv2 as cv


def print_proc(proc):
    tx = ''
    for num, el in enumerate(proc, start=0):
        tx += f'{num} - ({round(el, 3)})\n'
    return tx


def predict(net, img_name: str):
    np_arr = cv.imread(img_name)  # shape [28, 28, 3]
    # print(f'{np_arr.shape=}')
    transform = model_cnn.get_transform()
    tr = transform(np_arr)
    input = tr.reshape(1, 3, 28, 28)
    # print(f'{input.shape=}')

    pred = net(input)
    # print(f'{pred=}')
    proc = [tn.item() for tn in pred[0]]
    pred = pred.argmax()
    return pred.item(), proc


if __name__ == '__main__':
    net = torch.load('./net/cnn_net_3ch.pth')
    pred, proc = predict(net, './img/predict/9-img_53.png')
    # pred, proc = predict(net, './img/predict/num-362.jpg')
    print_proc(proc)
    print(f'Ваше число - {pred}')
