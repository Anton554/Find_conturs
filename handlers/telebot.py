#
# Модуль автобота
#
import os
import tempfile
import ansamble
import model_cnn
from aiogram import Dispatcher, types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.utils.callback_data import CallbackData
import main
import torch
import img_proces
from predict_one import predict, print_proc

cb_avtobot = CallbackData('pref', 'action', 'step')


class StatusAvto(StatesGroup):
    set_num = State()


def register_handlers(dp: Dispatcher):
    """
    Регистрация хендлеров
    :param dp:
    :return:
    """
    dp.register_message_handler(start, commands=['start'], state="*")
    dp.register_message_handler(download_photo, content_types=["photo"], state="*")
    dp.register_callback_query_handler(retry_ph, cb_avtobot.filter(step=["retry_ph"]), state="*")
    dp.register_callback_query_handler(select_num, cb_avtobot.filter(step=["select_num"]), state="*")


def register_handlers_final(dp: Dispatcher):
    """
    Регистрация хендлеров final
    :param dp:
    :return:
    """
    dp.register_message_handler(handler_txt, content_types=['text'], state="*")


# commands=['start'], state="*"
async def start(message: types.Message, state: FSMContext):
    await message.answer("Привет! Это автобот. Я собираю фотографии для обучения нейронной сети.")
    await qw_num(message)


async def qw_num(message: types.Message):
    keyboard = types.InlineKeyboardMarkup()
    ls_action = list(range(0, 5))
    ls_btn = []
    for action in ls_action:
        btn = types.InlineKeyboardButton(text=action, callback_data=cb_avtobot.new(action=action, step='select_num'))
        ls_btn.append(btn)
    keyboard.add(*ls_btn)
    ls_action = list(range(5, 10))
    ls_btn = []
    for action in ls_action:
        btn = types.InlineKeyboardButton(text=action, callback_data=cb_avtobot.new(action=action, step='select_num'))
        ls_btn.append(btn)
    keyboard.add(*ls_btn)
    # question = 'Выберите цифру от 0 до 9'
    question = 'Сфотографируйте цифру 7'
    # await message.answer(text=question, reply_markup=keyboard)
    await message.answer(text=question)


# cb_avtobot.filter(step=["select_num"]), state="*"
async def select_num(call: types.CallbackQuery, callback_data: dict, state: FSMContext):
    # await call.message.answer(f'Вы ввели : {callback_data["action"]} ')
    if '7' in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
    # if callback_data["action"] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        await state.update_data(class_num=callback_data["action"])
        await call.message.answer('Теперь сфотографируйте эту цифру')
    else:
        await qw_num(call.message)


# content_types=["photo"], state="*"
async def download_photo(message: types.Message, state: FSMContext):
    # создаем временный файл
    fd, path = tempfile.mkstemp(suffix='.jpg', text=True, dir='./tmp')
    await state.update_data(path_photo=path)
    await state.update_data(fd_photo=fd)
    # сохраняем фото в каталоге
    await message.photo[-1].download(destination_file=path)
    await message.answer("Сохранение фото ...")
    # Обработка фото
    await pars_num(message, state)


async def pars_num(message: types.Message, state: FSMContext):
    state_dc = await state.get_data()
    # Обрабатываем фото и сохраняем в папке ./img
    try:
        ph = img_proces.pars_img('7', img_name=state_dc['path_photo'])
        await state.update_data(pt_ph=ph)
        photo = open(ph, 'rb')
        await main.bot.send_photo(chat_id=message.chat.id, photo=photo)
        # net = ansamble.AnNNet()
        # net = model_cnn.CNNNet()
        # net.load_state_dict(torch.load('./net/cnn_net_2.pth'))
        net = torch.load('./net/cnn_net6_7_9.pth')
        # net = torch.load('C:/Projects/IT/Python/Net_pytorch/net/cnn_net_3ch.pth')ans_net.pth
        pred, ver = predict(net, ph)
        ver = print_proc(ver)
        await message.answer(f'{ver}')
        await message.answer(f'Ваше число - {pred}')
    except:
        await message.answer("Контур не найден.")
    finally:
        # Удаление временного файла
        if state_dc.get('fd_photo', 0) > 0:
            await delTemFile(state_dc['fd_photo'], state_dc['path_photo'])
        # await state.finish()
        # 'Повторите?'
        await get_num(message)


async def get_num(message: types.Message, ):
    keyboard = types.InlineKeyboardMarkup()
    ls_action = ['Да', 'Нет']
    ls_btn = []
    for action in ls_action:
        btn = types.InlineKeyboardButton(text=action, callback_data=cb_avtobot.new(action=action, step='retry_ph'))
        ls_btn.append(btn)
    keyboard.add(*ls_btn)
    question = 'Сохранить?'
    # question = 'Повторите?'
    await message.answer(text=question, reply_markup=keyboard)


# cb_avtobot.filter(step=["retry_ph"]), state="*"
async def retry_ph(call: types.CallbackQuery, callback_data: dict, state: FSMContext):
    state_dc = await state.get_data()
    # обработчик -- подтверждение ввода номера
    if callback_data["action"] == 'Да':
        await state.finish()
        await qw_num(call.message)
    else:
        os.remove(state_dc['pt_ph'])
        await state.finish()
        # await call.message.answer('Приходите ещё :-)')
        await qw_num(call.message)

async def delTemFile(fd, path):
    # закрываем дескриптор файла
    os.close(fd)
    # уничтожаем файл
    os.unlink(path)


# content_types=['text'], state="*"
async def handler_txt(message: types.Message):
    await message.answer('Напишите /start')
