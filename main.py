import asyncio
import torch
import setup
import logging
from handlers import telebot
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
import os

bot = Bot(token=setup.token, parse_mode=types.ParseMode.HTML)
dp = Dispatcher(bot, storage=MemoryStorage())
dir_prog = os.path.dirname(os.path.abspath(__file__))

# Загрузка модели нейронной сети Yolov5_m
model_vr = torch.hub.load('ultralytics/yolov5', 'custom', dir_prog + os.sep + 'net/yolov5s_300ep_8bt_40v.pt', force_reload = True, device='cpu')
model_num = torch.hub.load('ultralytics/yolov5', 'custom', dir_prog + os.sep + 'net/yolov5_m_200ep_4bt.pt', device='cpu')
model_num.conf = 0.50
model_vr.conf = 0.60


async def main():
    logging.basicConfig(level=logging.INFO)
    telebot.register_handlers(dp)
    telebot.register_handlers_final(dp)
    await dp.start_polling()


if __name__ == '__main__':
    asyncio.run(main())
