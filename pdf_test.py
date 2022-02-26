import copy
import pprint

import torch
import cv2 as cv

# Model
# model = torch.hub.load('ultralytics/yolov5', 'custom', './models/yolov5_s_80ep.pt')
# model = torch.hub.load('ultralytics/yolov5', 'custom', './models/yolov5_m_200ep_4bt.pt') !!!
model_vr = torch.hub.load('ultralytics/yolov5', 'custom', './net/yolov5m_300ep_8bt_10v.pt', device='cpu')
model_num = torch.hub.load('ultralytics/yolov5', 'custom', './net/yolov5_m_200ep_4var.pt', device='cpu')
model_num.conf = 0.60
model_vr.conf = 0.60
# Images
img = './img/ege/1_num.jpg'  # or file, Path, PIL, OpenCV, numpy, list
# Inference
results_vr = model_vr(img, size=1024)
results_num = model_num(img, size=1024)
# Results
results_vr.print()  # or .show(), .save(), .crop(), .pandas(), etc.
results_num.print()  # or .show(), .save(), .crop(), .pandas(), etc.
# results.crop()

# Отрисовка контуров найденых цифр
np_arr = cv.imread(img)
np_arr_etl = copy.copy(np_arr)
cn_img = results_vr.xyxy[0].size()[0]
for n in range(cn_img):
    x1 = int(results_vr.xyxy[0][n:n + 1, :][0][0].item())
    y1 = int(results_vr.xyxy[0][n:n + 1, :][0][1].item())
    x2 = int(results_vr.xyxy[0][n:n + 1, :][0][2].item()) * 4
    y2 = int(results_vr.xyxy[0][n:n + 1, :][0][3].item())
    pred = round(results_vr.xyxy[0][n:n + 1, :][0][4].item() * 100, 2)
    cls = int(results_vr.xyxy[0][n:n + 1, :][0][5].item())
    # print(f"{x1=} {y1=} {x2=} {y2=} {pred=}% {cls=}")

    cv.rectangle(np_arr, (x1, y1), (x2, y2), (255, 50, 0), 1)

    # if cls == 0 and pred > 50:
    #     cv.rectangle(np_arr, (x1, y1), (x2, y2), (255, 50, 0), 1)
    # elif cls == 1 and pred > 30:
    #     cv.rectangle(np_arr, (x1, y1), (x2, y2), (0, 0, 255), 1)
    # elif cls > 1 and pred > 50:
    #     cv.rectangle(np_arr, (x1, y1), (x2, y2), (10, 255, 10), 1)


def find_center(res, np_arr):
    """ Возвращает список объектов num and no_num

    :param res: Тензор с координатами
    :param np_arr: Np_mas массив изображения
    :return: [координата х, координата у, класификация, np_mas изображения]
    """
    rez = []
    for tn in res.xyxy[0]:
        ls = [el.item() for el in tn]
        if ls[-1] == 1 or ls[-1] == 0:
            x = (ls[3] - ls[1]) / 2 + ls[1]
            y = (ls[2] - ls[0]) / 2 + ls[0]
            np = np_arr[int(ls[1]):int(ls[3]), int(ls[0]):int(ls[2]), :]
            rez.append([x, y, int(ls[-1]), np])
    return rez


def my_fn(results_vr):
    """

    :param results_vr:
    :return:
    """
    dc = {}
    cn_img = results_vr.xyxy[0].size()[0]
    for n in range(cn_img):
        x1 = int(results_vr.xyxy[0][n:n + 1, :][0][0].item())
        y1 = int(results_vr.xyxy[0][n:n + 1, :][0][1].item())
        x2 = int(results_vr.xyxy[0][n:n + 1, :][0][2].item()) * 4
        y2 = int(results_vr.xyxy[0][n:n + 1, :][0][3].item())
        num = int(results_vr.xyxy[0][n:n + 1, :][0][5].item())
        # pred = round(results_vr.xyxy[0][n:n + 1, :][0][4].item() * 100, 2)
        dc[num + 1] = [x1, y1, x2, y2]
    return dc


def my_fn2(num_v, ls_obj):
    """

    :param num_v:
    :param ls_obj:
    :return:
    """
    dc = {}
    ls = []
    ls_obj = sorted(ls_obj, key=lambda x: x[1])
    for coord in num_v.items():
        for el in ls_obj:
            if (coord[1][0] <= el[1] and coord[1][1] <= el[0]) and (coord[1][2] >= el[1] and coord[1][3] >= el[0]):
                ls.extend([el[-1], el[2]])
                cv.imshow('Win_img', el[-1])
                cv.waitKey(0)
        dc[coord[0]] = [ls]
        ls = []
    return dc


if __name__ == '__main__':
    cv.imwrite('result.jpg', np_arr)
    ls_obj = find_center(results_num, np_arr_etl)
    obj_vr = my_fn(results_vr)
    pprint.pprint(my_fn2(obj_vr, ls_obj))

# cv.imshow('Win_img', np_arr)
# cv.waitKey(0)
# cv.destroyWindow('Win_img')
# print(results.xyxy[0])
