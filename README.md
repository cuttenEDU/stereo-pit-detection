# stereo-pit-detection
Обнаружение автодорожных выбоин на стерео-изображении / Driveway pit detection on stereo-images
<br/><br/>
# RUS
## Системные требования
  +	Операционная система Ubuntu Linux версии 18.05.4 LTS
  +	Интерпретатор Python версии 3.7
  +	Фреймворк PyTorch версии 1.8.1
  +	Платформа CUDA версии 10.2
  +	Библиотека Detectron2 версии 2.0.4
   
  Дополнительные требования к установленным модулям языка Python расположены в файле requirements.txt

  
## Запуск
Каждый скрипт (кроме MC-CNN/model.py) имеет интерфейс командной строки. Для вызова справки об интерфейсе достаточно запустить скрипт без всяких аргументов командной строки.

## Обученые модели
MC-CNN: https://drive.google.com/file/d/1riH1ELszBPzKKVa1BYzoVleavsqrZkp0/view?usp=sharing  
MASK R-CNN: https://drive.google.com/file/d/1LPRd46q7QNyFyMMu5lnVK4ZUhiSu3yGm/view?usp=sharing

<br/><br/>
# ENG
## System requirements
  +	Ubuntu Linux 18.05.4 LTS
  +	Python 3.7
  +	PyTorch 1.8.1
  +	CUDA 10.2
  +	Detectron2 2.0.4
  
  List of needed python modules you can find in requirements.txt
  
## Launching scripts
Each .py file (excluding MC-CNN/model.py) has CLI interface. To get help on how to use it, simply launch the script without any arguments.

## Pretrained models
MC-CNN: https://drive.google.com/file/d/1riH1ELszBPzKKVa1BYzoVleavsqrZkp0/view?usp=sharing  
MASK R-CNN: https://drive.google.com/file/d/1LPRd46q7QNyFyMMu5lnVK4ZUhiSu3yGm/view?usp=sharing
