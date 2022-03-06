"""
Библиотека функций для детектирования объектов на бланке ответов ЕГЭ
"""
import pprint
from img_proces import conv_img_pdf
import cv2 as cv
from predict_one import predict


# Images
# img = './img/ege/2_num.jpg'  # or file, Path, PIL, OpenCV, numpy, list
# Inference
# Results
# results_vr.print()  # or .show(), .save(), .crop(), .pandas(), etc.
# results_num.print()  # or .show(), .save(), .crop(), .pandas(), etc.
# results.crop()

# Отрисовка контуров найденых цифр
"""np_arr = cv.imread(img)
np_arr_etl = copy.copy(np_arr)
cn_img = results_vr.xyxy[0].size()[0]
for n in range(cn_img):
    x1 = int(results_vr.xyxy[0][n:n + 1, :][0][0].item())
    y1 = int(results_vr.xyxy[0][n:n + 1, :][0][1].item())
    x2 = int(results_vr.xyxy[0][n:n + 1, :][0][2].item())
    x2 = x2 + (x2 - x1) * 7
    y2 = int(results_vr.xyxy[0][n:n + 1, :][0][3].item())
    pred = round(results_vr.xyxy[0][n:n + 1, :][0][4].item() * 100, 2)
    cls = int(results_vr.xyxy[0][n:n + 1, :][0][5].item())
    # print(f"{x1=} {y1=} {x2=} {y2=} {pred=}% {cls=}")

    cv.rectangle(np_arr, (x1, y1), (x2, y2), (255, 50, 0), 1)"""


def find_center(res, np_arr):
    """ Возвращает список объектов num and no_num

    :param res: Тензор с координатами
    :param np_arr: Np_mas массив изображения
    :return: [[координата х, координата у, класификация, np_mas изображения],
     [координата х, координата у, класификация, np_mas изображения]]
    """
    rez = []
    for tn in res.xyxy[0]:
        ls = [el.item() for el in tn]
        if ls[-1] == 1 or ls[-1] == 0:
            x = (ls[3] - ls[1]) / 2 + ls[1]
            y = (ls[2] - ls[0]) / 2 + ls[0]
            np = np_arr[int(ls[1]):int(ls[3]), int(ls[0]):int(ls[2]), :]
            rez.append([x, y, int(ls[-1]), np])
    rez = remov(res, np_arr, rez)
    return rez


def remov(res, np_arr, ls: list):
    """Убирает ошибочные символы
    Принимает тензор с данными и список объектов num and no_num
    типа [[координата х, координата у, класификация, np_mas изображения],
    [координата х, координата у, класификация, np_mas изображения]].
    Возвращает список объектов num and no_num в котором отсутствуют ошибочные символы.

    :param res: Тензор из нейроки
    :param np_arr: Np_mas массив изображения
    :param ls: [[координата х, координата у, класификация, np_mas изображения],
     [координата х, координата у, класификация, np_mas изображения]]
    :return: [[координата х, координата у, класификация, np_mas изображения],
     [координата х, координата у, класификация, np_mas изображения]]
    """
    rez_dc = {}
    fin_ls = []
    ls_v = []
    for el in ls:
        for tn in res.xyxy[0]:
            coord = [el.item() for el in tn]
            if (coord[0] <= el[1] and coord[1] <= el[0]) and (coord[2] >= el[1] and coord[3] >= el[0]):
                ls_v.append(coord)
        # Список из элементов с пересмкающимися координатами
        # [[[281.5054016113281, 983.6282958984375, 309.0924377441406, 1022.3548583984375, 0.811305582523346, 0.0]],
        # [[280.5054016113281, 981.6282958984375, 308.0924377441406, 1012.3548583984375, 0.511305582523346, 1.0]]]
        ls_v = sorted(ls_v, key=lambda x: x[4])[-1]
        s = ''.join(list(map(str, ls_v)))
        rez_dc[s] = ls_v
        ls_v = []
    rez = [el for el in rez_dc.values()]
    for ls in rez:
        if ls[-1] == 1 or ls[-1] == 0:
            x = (ls[3] - ls[1]) / 2 + ls[1]
            y = (ls[2] - ls[0]) / 2 + ls[0]
            np = np_arr[int(ls[1]):int(ls[3]), int(ls[0]):int(ls[2]), :]
            fin_ls.append([x, y, int(ls[-1]), np])
    return fin_ls


def find_coord(results_vr):
    """Находит крайнии точки варианта
    Принимает тензор с данными, возвращает словарь номер: координаты

    :param results_vr: Тензор из нейроки
    :return: Словарь типа {номер: [x1, y1, x2, y2]}
    """
    dc = {}
    cn_img = results_vr.xyxy[0].size()[0]
    for n in range(cn_img):
        x1 = int(results_vr.xyxy[0][n:n + 1, :][0][0].item())
        y1 = int(results_vr.xyxy[0][n:n + 1, :][0][1].item())
        x2 = int(results_vr.xyxy[0][n:n + 1, :][0][2].item())
        x2 = x2 + (x2 - x1) * 7
        y2 = int(results_vr.xyxy[0][n:n + 1, :][0][3].item())
        dc[n + 1] = [x1, y1, x2, y2]
    return dc


def real_num(dc: dict):
    """Присваивает варианту настоящий порядковый номер
    Принимает словарь с ошибочными порядковыми номерами.
    Возвращает словарь с правильными порядковыми номерами.

    :param dc: Словарь типа {ложный номер: [x1, y1, x2, y2]}
    :return: Словарь типа {порядковый номер: [x1, y1, x2, y2]}
    """
    ls_left = []
    ls_right = []
    ls_fin = []
    dc_fin = {}
    for el in dc.values():
        if el[0] < 300:
            ls_left.append(el)
        else:
            ls_right.append(el)

    ls_left = sorted(ls_left, key=lambda x: x[1])
    ls_right = sorted(ls_right, key=lambda x: x[1])
    ls_fin.extend(ls_left)
    ls_fin.extend(ls_right)
    for n in range(len(ls_fin)):
        dc_fin[n + 1] = ls_fin[n]
    return dc_fin


def relate_num(num_v: dict, ls_ans: list):
    """Соотносит варианты и ответы на задания
    Принимает словарь вариантов и список объектов num and no_num, возвращает изображение объекта num and no_num
    находящегося вдутри границ варианта и значение его класса - (cls) no_num -> 1 / num -> 0

    :param num_v: Словарь {номер: [x, y, x, y]}
    :param ls_ans: Список ответов
    :return: <class 'dict'> вида {номер: [[np_arr, cls], [np_arr, cls]]}
    """
    dc = {}
    ls = []
    ls_ans = sorted(ls_ans, key=lambda x: x[1])
    for coord in num_v.items():
        for el in ls_ans:
            if (coord[1][0] <= el[1] and coord[1][1] <= el[0]) and (coord[1][2] >= el[1] and coord[1][3] >= el[0]):
                ls.append([el[-1], el[2]])
                # cv.imshow('Win_img', el[-1])
                # cv.waitKey(0)
        dc[coord[0]] = [ls]
        ls = []
    return dc


def detekt(dc_obj: dict, net):
    """

    :param dc_obj: <class 'dict'> вида {1: [[np_arr, cls], [np_arr, cls]]}
    :return: <class 'dict'> вида {1: '-0,7'}
    """
    dc = {}
    for el in dc_obj.items():
        st = raspozn(el[1], net)
        dc[el[0]] = st
    return dc


def raspozn(ls: list, net):
    """

    :param ls: [[np_arr, cls], [np_arr, cls]]
    :return: <class 'str'> вида '-0,7'
    """
    st = ''
    for num, el in enumerate(ls[0], start=1):
        if num == 1 and el[1] == 1:
            st += '-'
        elif el[1] == 1:
            st += ','
        else:
            img = conv_img_pdf(el[0])
            cv.imwrite(f'crop_img.png', img)
            pred, ver = predict(net, 'crop_img.png')
            st += str(pred)
    return st


def pdf_start(results_num, results_vr, img, net):
    np_arr_etl = cv.imread(img)
    ls_obj = find_center(results_num, np_arr_etl)
    obj_vr = find_coord(results_vr)
    obj_vr = real_num(obj_vr)
    dc_obj = relate_num(obj_vr, ls_obj)
    pprint.pprint(detekt(dc_obj, net))

if __name__ == '__main__':
    pass
    # cv.imwrite('result.jpg', np_arr)
    # ls_obj = find_center(results_num, np_arr_etl)
    # obj_vr = find_coord(results_vr)
    # obj_vr = real_num(obj_vr)
    # dc_obj = relate_num(obj_vr, ls_obj)
    # pprint.pprint(detekt(dc_obj))
# cv.imshow('Win_img', np_arr)
# cv.waitKey(0)
# cv.destroyWindow('Win_img')
# print(results.xyxy[0])
