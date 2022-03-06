import torch
import model_cnn
import cv2 as cv


def print_proc(proc):
    tx = ''
    for num, el in enumerate(proc, start=0):
        tx += f'{num} - {round(el * 100, 3)} %\n'
    return tx


def print_proc_fin(proc):
    return round(max(proc) * 100, 3)


def predict(net, img_name: str):
    np_arr = cv.imread(img_name)  # shape [28, 28, 3]
    transform = model_cnn.get_transform()
    tr = transform(np_arr)
    input = tr.reshape(1, 3, 28, 28)
    pred = net(input)
    pred = torch.softmax(pred, dim=1)
    proc = [tn.item() for tn in pred[0]]
    pred = pred.argmax()
    return pred.item(), proc
"""tensor([[[[-1., -1., -1.,  ..., -1., -1., -1.],
          [-1., -1., -1.,  ..., -1., -1., -1.],
          [-1., -1., -1.,  ..., -1., -1., -1.],
          ...,
          [-1., -1., -1.,  ..., -1., -1., -1.],
          [-1., -1., -1.,  ..., -1., -1., -1.],
          [-1., -1., -1.,  ..., -1., -1., -1.]],

         [[-1., -1., -1.,  ..., -1., -1., -1.],
          [-1., -1., -1.,  ..., -1., -1., -1.],
          [-1., -1., -1.,  ..., -1., -1., -1.],
          ...,
          [-1., -1., -1.,  ..., -1., -1., -1.],
          [-1., -1., -1.,  ..., -1., -1., -1.],
          [-1., -1., -1.,  ..., -1., -1., -1.]],

         [[-1., -1., -1.,  ..., -1., -1., -1.],
          [-1., -1., -1.,  ..., -1., -1., -1.],
          [-1., -1., -1.,  ..., -1., -1., -1.],
          ...,
          [-1., -1., -1.,  ..., -1., -1., -1.],
          [-1., -1., -1.,  ..., -1., -1., -1.],
          [-1., -1., -1.,  ..., -1., -1., -1.]]]])"""





if __name__ == '__main__':
    net = torch.load('C:/Projects/IT/Python/Find_conturs/net/cnn_net6_7_9_97.pth')
    pred, proc = predict(net, './img/fin/8-img_51.png')
    # pred, proc = predict(net, './img/predict/num-362.jpg')
    print(print_proc(proc))
    print(f'Ваше число - {pred}')
