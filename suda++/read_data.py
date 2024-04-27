# -*- coding: utf-8 -*-
import os.path
from datetime import datetime
import serial
import numpy as np

ser = serial.Serial("COM5", 115200, timeout=0.5)  # ֻҪ�Ĵ��ںž���
file_name = 'tt111111.txt'

# ���ǻ��⴫���ݾ����api��������ʽ��??
def receive_data_thread():
    ser.readline() # �ն�һ�����ݷ�ֹbyteת���쳣
    line = ''
    # �����Ҫ������յ����ݷ�����ԣ�������������ע��&whileѭ��������һ��write_file����õ�ע��
    if os.path.exists(file_name):
        print('file exist')
        # return
    write_file = open(file_name, 'w')
    while True:
        temp = ser.readline()
        line += str(temp, encoding="utf-8")
        if len(line) == 0 or line[-1] != '\n':  # һ�����ݿ���û������жϣ�������line��¼ֱ�ӳ���\n�ٴ�??
            continue
        # �����Ҫ������յ����ݷ�����ԣ�������������ע��
        write_file.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + ',' + str(temp, encoding="utf-8")[:-1])
        print(line)  # ���γ�ȥ�������
        points =  [int(x) for x in line.split(',')[:2]]
        line = ''
        
        
        
def read_data():
    temp = ser.readline()        
    line = str(temp, encoding="utf-8")
    while len(line) == 0 or line[-1] !='\n':
        temp = ser.readline()        
        line = str(temp, encoding="utf-8")
    points = [int(x) for x in line.split(',')[:2]]
    return points

# receive_data_thread()
