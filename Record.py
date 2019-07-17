# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 15:43:30 2019

@author: jack
"""

import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from directkeys import PressKey, ReleaseKey, W, A, S, D
from grab import grab_screen
from getkeys import  key_check
import math
import os
from PyQt5 import QtCore, QtGui,QtWidgets
import resourse_rc
import pandas as pd
from random import shuffle
from  alexnet import alex
from googlenet import lenet
from keras.models import Sequential
from keras.models import load_model
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
height = 120
def output_key(keys):
      output = [0,0,0]
      if "A" in keys:
            output[0] = 1
      elif 'D' in keys:
            output[2] = 1
      else:
            output[1] = 1
      return output

file_name = 'training_data.npy'

if os.path.isfile(file_name):
      print('data already loaded')
      training_data = list(np.load(file_name))
else:
      print('no training data')
      training_data = []

            
class Ui_MainWindow(object):

    def record(self):
            start = time.clock()
            ellapsed = 0
            paused = 0
            st = ""
            while(True):
                QtWidgets.QApplication.processEvents()
                if self.pause_3.isChecked():
                  paused = (time.clock() - ellapsed)
                  continue
                
                screen =  grab_screen(region = (40, 80, 800, 600))
                screen  = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
                screen =  cv2.resize(screen,(height,height))
                keys = key_check()
                output = output_key(keys)
                training_data.append([screen,output])
                ellapsed = int((time.clock() - start)-paused)
                st = str("training_data for about : "+ str(ellapsed)+"\n")
                print(st)
                self.display.display(ellapsed)
                  #cv2.imshow('my_driver_bot',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
                if len(training_data)%self.batch_slider.value() == 0:
                        np.save(file_name, training_data)
                if self.stop_3.isChecked():
                        cv2.destroyAllWindows()
                        break
                
    def processed(self):
      
      df = pd.DataFrame(training_data)
      lefts = []
      rights = []
      straight = []

      shuffle(training_data)

      for data in training_data:
            QtWidgets.QApplication.processEvents()
            img = data[0]
            action = data[1]
            
            if action == [1,0,0]:
                  lefts.append([img, action])
            
            elif action == [0,1,0]:
                  straight.append([img, action])
            elif action == [0,0,1]:
                  rights.append([img, action])
            else:
                  print('no values')

      length = min(len(lefts),len(straight),len(rights))
      straight = straight[:length]
      lefts = lefts[:length]
      rights = rights[:length]

      final_data = straight+lefts+rights
      self.length.setText(str(len(final_data)))
      self.count_status.setText("process sucessful!!")
      np.save('final_train',final_data)





    def samp(self):
        train_data = np.load('final_train.npy')
        if train_data.shape != (0,):
            plt.imshow(train_data[1][0])
            plt.show()      
        else:
                self.count_status.setText("nothing to show")

    def tr(self):
        QtWidgets.QApplication.processEvents()
        self.progressBar.setValue(0)
        epoch = self.epo.value()
        lr = 0.1/self.learningrt.value()
        name = alex
        
        model = alex(height,height)
        if self.alexnetwork.isChecked():
            name = alex
            model = alex(height,height)
        elif self.googlenetwork.isChecked():
            name = lenet
            model = lenet(height,height,lr)
        model_name = "self-driving-car"
        train_data = np.load('final_train.npy')
        shuffle(train_data)
        x = np.array([i[0] for i in train_data]).reshape(-1,height,height,1)
        y = np.array([i[1] for i in train_data])
        
        los =0
        acc =0
        if self.alexnetwork.isChecked():
            los, acc = model.evaluate(x,y)
        else:
            acc = model.evaluate(x,y)

        num = 0
        x_train, x_test,y_train, y_test = train_test_split(x,y, test_size =0.02, random_state= 42 )
        for e in range(epoch):
            QtWidgets.QApplication.processEvents()
            num += 100/epoch
            if self.alexnetwork.isChecked():
                model.fit(x_train,y_train, batch_size=32,shuffle=True, validation_data=(x_test, y_test))
            else:
                model.fit(x_train,y_train, batch_size=32,shuffle=True, validation_set=(x_test, y_test),n_epoch= 1)
            

            self.progressBar.setValue(num)
            self.loss.setText("")
            self.loss.setText(str(los))
            self.accuracy.setText("")
            self.accuracy.setText(str(acc))

            
        model.save(model_name)
    
    def rec_page(self):
        self.stackedWidget.setCurrentIndex(0)
    def procc_page(self):
        self.stackedWidget.setCurrentIndex(1)
    def train_page(self):
        self.stackedWidget.setCurrentIndex(2)
    def test_page(self):
        self.stackedWidget.setCurrentIndex(3)

    
    def tes(self):
        
        model_name = "self-driving-car"
        model = alex(height,height)
        model = load_model(model_name)
        sleep = 0.2
        def straight():
            PressKey(W)
            ReleaseKey(A)
            ReleaseKey(D)


        def left():
            PressKey(A)
            PressKey(W)
            ReleaseKey(D)
            time.sleep(sleep)
            ReleaseKey(A)




        def right():
            PressKey(S)
            PressKey(W)
            ReleaseKey(A)
            time.sleep(sleep)
            ReleaseKey(S)



        while(True):
            QtWidgets.QApplication.processEvents()
            if self.pause_4.isChecked():
                continue
            screen =  grab_screen(region = (40, 80, 800, 640))
            screen  = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen =  cv2.resize(screen,(height,height))

            predict = model.predict([screen.reshape(1,height,height,1)])
            predict = predict[0]
            moves = list(np.around(predict))
            
            if moves ==[1,0,0]:
                left()
                self.feedback.setPlainText("\n Turn left")
            elif moves ==[0,1,0]:
                straight()
                self.feedback.setPlainText("\n Go straight")
            elif moves == [0,0,1]:
                right()
                self.feedback.setPlainText("\n Turn Right")


            keys = key_check()

            if self.stop_4.isChecked():
                cv2.destroyAllWindows()
                break



    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(601, 312)
        MainWindow.setMinimumSize(QtCore.QSize(601, 312))
        MainWindow.setMaximumSize(QtCore.QSize(601, 312))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("images/01_Big_File_computer_internet_file_data-512.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet("background-color:rgb(47, 54, 64)")
        MainWindow.setDocumentMode(False)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Triangular)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setAutoFillBackground(False)
        self.centralwidget.setStyleSheet("")
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(0, 0, 601, 291))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.frame.setPalette(palette)
        self.frame.setStyleSheet("background-image:url(:/newPrefix/back.png)")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame.setLineWidth(0)
        self.frame.setObjectName("frame")
        self.stackedWidget = QtWidgets.QStackedWidget(self.frame)
        self.stackedWidget.setGeometry(QtCore.QRect(0, 0, 601, 291))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.stackedWidget.setFont(font)
        self.stackedWidget.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.stackedWidget.setFocusPolicy(QtCore.Qt.TabFocus)
        self.stackedWidget.setToolTipDuration(-1)
        self.stackedWidget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.stackedWidget.setAutoFillBackground(False)
        self.stackedWidget.setStyleSheet("background-image:url(:/newPrefix/back.jpg)")
        self.stackedWidget.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.stackedWidget.setObjectName("stackedWidget")
        
        self.page1 = QtWidgets.QWidget()
        self.page1.setObjectName("page1")
        self.frame_2 = QtWidgets.QFrame(self.page1)
        self.frame_2.setGeometry(QtCore.QRect(0, -10, 601, 321))
        self.frame_2.setStyleSheet("background-image:url(:/newPrefix/back.jpg)")
        self.frame_2.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setLineWidth(0)
        self.frame_2.setObjectName("frame_2")
        self.label = QtWidgets.QLabel(self.frame_2)
        self.label.setGeometry(QtCore.QRect(200, 140, 71, 41))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.label.setPalette(palette)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.run = QtWidgets.QPushButton(self.frame_2)
        self.run.setGeometry(QtCore.QRect(30, 70, 141, 51))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(245, 246, 250))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(68, 189, 50))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(245, 246, 250))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(245, 246, 250))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(68, 189, 50))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(68, 189, 50))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(245, 246, 250))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(68, 189, 50))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(245, 246, 250))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(245, 246, 250))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(68, 189, 50))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(68, 189, 50))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(245, 246, 250))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(68, 189, 50))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(245, 246, 250))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(245, 246, 250))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(68, 189, 50))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(68, 189, 50))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.run.setPalette(palette)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.run.setFont(font)
        self.run.setStyleSheet("background-color: rgb(68, 189, 50);\n"
"color: rgb(245, 246, 250)")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("images/_Record-512.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.run.setIcon(icon1)
        self.run.setIconSize(QtCore.QSize(40, 40))
        self.run.setObjectName("run")
        self.display = QtWidgets.QLCDNumber(self.frame_2)
        self.display.setGeometry(QtCore.QRect(20, 140, 171, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.display.setFont(font)
        self.display.setStyleSheet("color : rgb(76, 209, 55)\n"
"")
        self.display.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.display.setSmallDecimalPoint(True)
        self.display.setDigitCount(7)
        self.display.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.display.setObjectName("display")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.frame_2)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(400, 30, 171, 191))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_3.setContentsMargins(1, 1, 1, 1)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.stop_3 = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(232, 65, 24))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(232, 65, 24))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(232, 65, 24))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(232, 65, 24))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(232, 65, 24))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(232, 65, 24))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(232, 65, 24))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(232, 65, 24))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(232, 65, 24))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.stop_3.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Microsoft New Tai Lue")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.stop_3.setFont(font)
        self.stop_3.setStyleSheet("color: rgb(232, 65, 24);\n"
"background-color:transparent")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("images/media_player_button_11-512.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.stop_3.setIcon(icon2)
        self.stop_3.setIconSize(QtCore.QSize(20, 20))
        self.stop_3.setObjectName("stop_3")
        self.verticalLayout_3.addWidget(self.stop_3)
        self.pause_3 = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 151, 230))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 151, 230))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 151, 230))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 151, 230))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 151, 230))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 151, 230))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 151, 230))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 151, 230))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 151, 230))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.pause_3.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Microsoft New Tai Lue")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.pause_3.setFont(font)
        self.pause_3.setStyleSheet("color: rgb(0, 151, 230)\n"
"")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("images/pause_media_multimedia_icon_video_blue_flat-512.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pause_3.setIcon(icon3)
        self.pause_3.setIconSize(QtCore.QSize(20, 20))
        self.pause_3.setObjectName("pause_3")
        self.verticalLayout_3.addWidget(self.pause_3)
        self.continue_4 = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(76, 209, 55))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(76, 209, 55))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(76, 209, 55))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(76, 209, 55))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(76, 209, 55))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(76, 209, 55))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(76, 209, 55))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(76, 209, 55))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(76, 209, 55))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.continue_4.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Microsoft New Tai Lue")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.continue_4.setFont(font)
        self.continue_4.setStyleSheet("color: rgb(76, 209, 55)\n"
"")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("images/13-512.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.continue_4.setIcon(icon4)
        self.continue_4.setIconSize(QtCore.QSize(20, 20))
        self.continue_4.setChecked(True)
        self.continue_4.setObjectName("continue_4")
        self.verticalLayout_3.addWidget(self.continue_4)
        self.batch_slider = QtWidgets.QSlider(self.frame_2)
        self.batch_slider.setGeometry(QtCore.QRect(30, 210, 201, 41))
        self.batch_slider.setStyleSheet("QSlider::groove:horizontal {\n"
"border: 1px solid #999999;\n"
"height: 18px;\n"
"\n"
"border-radius: 9px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"width: 18px;\n"
" background-image: url(:/slider-knob.png)\n"
"}\n"
"\n"
"QSlider::add-page:qlineargradient {\n"
"background: lightgrey;\n"
"border-top-right-radius: 9px;\n"
"border-bottom-right-radius: 9px;\n"
"border-top-left-radius: 0px;\n"
"border-bottom-left-radius: 0px;\n"
"}\n"
"\n"
"QSlider::sub-page:qlineargradient {\n"
"background: rgb(235, 77, 75);\n"
"border-top-right-radius: 0px;\n"
"border-bottom-right-radius: 0px;\n"
"border-top-left-radius: 9px;\n"
"border-bottom-left-radius: 9px;\n"
"}")
        self.batch_slider.setMinimum(100)
        self.batch_slider.setMaximum(10000)
        self.batch_slider.setOrientation(QtCore.Qt.Horizontal)
        self.batch_slider.setObjectName("batch_slider")
        self.label_2 = QtWidgets.QLabel(self.frame_2)
        self.label_2.setGeometry(QtCore.QRect(240, 210, 91, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color: rgb(250, 130, 49)\n"
"")
        self.label_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label_2.setObjectName("label_2")
        self.line_2 = QtWidgets.QFrame(self.frame_2)
        self.line_2.setGeometry(QtCore.QRect(343, 10, 20, 291))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.next1 = QtWidgets.QPushButton(self.frame_2)
        self.next1.setGeometry(QtCore.QRect(500, 250, 81, 41))
        self.next1.setStyleSheet("color: rgb(15, 185, 177);\n"
"background-color: rgb(15, 185, 177)")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("images/30-512.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.next1.setIcon(icon5)
        self.next1.setIconSize(QtCore.QSize(20, 20))
        self.next1.setObjectName("next1")
        self.stackedWidget.addWidget(self.page1)
        self.page2 = QtWidgets.QWidget()
        self.page2.setObjectName("page2")
        self.frame_3 = QtWidgets.QFrame(self.page2)
        self.frame_3.setGeometry(QtCore.QRect(0, 0, 601, 311))
        self.frame_3.setStyleSheet("background-image:url(:/newPrefix/back.jpg)")
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.process = QtWidgets.QPushButton(self.frame_3)
        self.process.setGeometry(QtCore.QRect(70, 30, 161, 71))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.process.setFont(font)
        self.process.setStyleSheet("color: rgb(245, 246, 250);\n"
"background-color: rgb(156, 136, 255)")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("images/037_-_File_Processing-512.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.process.setIcon(icon6)
        self.process.setIconSize(QtCore.QSize(50, 50))
        self.process.setObjectName("process")
        self.label_3 = QtWidgets.QLabel(self.frame_3)
        self.label_3.setGeometry(QtCore.QRect(70, 150, 101, 31))
        font = QtGui.QFont()
        font.setFamily("MS Reference Sans Serif")
        font.setPointSize(12)
        font.setItalic(False)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("color: rgb(165, 94, 234)\n"
"")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.frame_3)
        self.label_4.setGeometry(QtCore.QRect(70, 200, 101, 31))
        font = QtGui.QFont()
        font.setFamily("MS Reference Sans Serif")
        font.setPointSize(12)
        font.setItalic(False)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("color: rgb(38, 222, 129)")
        self.label_4.setObjectName("label_4")
        self.sample = QtWidgets.QPushButton(self.frame_3)
        self.sample.setGeometry(QtCore.QRect(300, 30, 161, 71))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.sample.setFont(font)
        self.sample.setStyleSheet("color: rgb(245, 246, 250);\n"
"background-color: rgb(76, 209, 55)")
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("images/camera.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.sample.setIcon(icon7)
        self.sample.setIconSize(QtCore.QSize(50, 50))
        self.sample.setObjectName("sample")
        self.count_status = QtWidgets.QLabel(self.frame_3)
        self.count_status.setGeometry(QtCore.QRect(190, 150, 161, 31))
        font = QtGui.QFont()
        font.setFamily("MS Reference Sans Serif")
        font.setPointSize(10)
        font.setItalic(False)
        self.count_status.setFont(font)
        self.count_status.setStyleSheet("color: rgb(165, 94, 234)\n"
"")
        self.count_status.setText("")
        self.count_status.setObjectName("count_status")
        self.length = QtWidgets.QLabel(self.frame_3)
        self.length.setGeometry(QtCore.QRect(190, 200, 161, 31))
        font = QtGui.QFont()
        font.setFamily("MS Reference Sans Serif")
        font.setPointSize(12)
        font.setItalic(False)
        self.length.setFont(font)
        self.length.setStyleSheet("color: rgb(38, 222, 129)")
        self.length.setText("")
        self.length.setObjectName("length")
        self.next2 = QtWidgets.QPushButton(self.frame_3)
        self.next2.setGeometry(QtCore.QRect(500, 240, 81, 41))
        self.next2.setStyleSheet("color: rgb(15, 185, 177);\n"
"background-color: rgb(15, 185, 177)")
        self.next2.setIcon(icon5)
        self.next2.setIconSize(QtCore.QSize(20, 20))
        self.next2.setObjectName("next2")
        self.stackedWidget.addWidget(self.page2)
        self.page3 = QtWidgets.QWidget()
        self.page3.setObjectName("page3")
        self.frame_4 = QtWidgets.QFrame(self.page3)
        self.frame_4.setGeometry(QtCore.QRect(0, 0, 601, 291))
        self.frame_4.setStyleSheet("background-image:url(:/newPrefix/back.jpg)")
        self.frame_4.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.frame_5 = QtWidgets.QFrame(self.frame_4)
        self.frame_5.setGeometry(QtCore.QRect(0, 0, 181, 141))
        self.frame_5.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.frame_5)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(10, 20, 152, 111))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.alexnetwork = QtWidgets.QRadioButton(self.verticalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.alexnetwork.setFont(font)
        self.alexnetwork.setStyleSheet("color: rgb(72, 126, 176)")
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap("images/neural-network-3-501246.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.alexnetwork.setIcon(icon8)
        self.alexnetwork.setIconSize(QtCore.QSize(30, 30))
        self.alexnetwork.setChecked(True)
        self.alexnetwork.setObjectName("alexnetwork")
        self.verticalLayout_4.addWidget(self.alexnetwork)
        self.googlenetwork = QtWidgets.QRadioButton(self.verticalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.googlenetwork.setFont(font)
        self.googlenetwork.setStyleSheet("color: rgb(251, 197, 49)")
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap("images/index.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.googlenetwork.setIcon(icon9)
        self.googlenetwork.setIconSize(QtCore.QSize(30, 30))
        self.googlenetwork.setObjectName("googlenetwork")
        self.verticalLayout_4.addWidget(self.googlenetwork)
        self.verticalLayoutWidget_2.raise_()
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.frame_4)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(230, 20, 191, 111))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.learningrt = QtWidgets.QSlider(self.verticalLayoutWidget_3)
        self.learningrt.setStyleSheet("QSlider::groove:horizontal {\n"
"border: 1px solid #999999;\n"
"height: 18px;\n"
"\n"
"border-radius: 9px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"width: 18px;\n"
" background-image: url(:/slider-knob.png)\n"
"}\n"
"\n"
"QSlider::add-page:qlineargradient {\n"
"background: lightgrey;\n"
"border-top-right-radius: 9px;\n"
"border-bottom-right-radius: 9px;\n"
"border-top-left-radius: 0px;\n"
"border-bottom-left-radius: 0px;\n"
"}\n"
"\n"
"QSlider::sub-page:qlineargradient {\n"
"background:rgb(106, 176, 76);\n"
"border-top-right-radius: 0px;\n"
"border-bottom-right-radius: 0px;\n"
"border-top-left-radius: 9px;\n"
"border-bottom-left-radius: 9px;\n"
"}")
        self.learningrt.setMinimum(1)
        self.learningrt.setMaximum(15)
        self.learningrt.setOrientation(QtCore.Qt.Horizontal)
        self.learningrt.setObjectName("learningrt")
        self.verticalLayout.addWidget(self.learningrt)
        self.epo = QtWidgets.QSlider(self.verticalLayoutWidget_3)
        self.epo.setStyleSheet("QSlider::groove:horizontal {\n"
"border: 1px solid #999999;\n"
"height: 18px;\n"
"\n"
"border-radius: 9px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"width: 18px;\n"
" background-image: url(:/slider-knob.png)\n"
"}\n"
"\n"
"QSlider::add-page:qlineargradient {\n"
"background: lightgrey;\n"
"border-top-right-radius: 9px;\n"
"border-bottom-right-radius: 9px;\n"
"border-top-left-radius: 0px;\n"
"border-bottom-left-radius: 0px;\n"
"}\n"
"\n"
"QSlider::sub-page:qlineargradient {\n"
"background: rgb(255, 121, 121);\n"
"border-top-right-radius: 0px;\n"
"border-bottom-right-radius: 0px;\n"
"border-top-left-radius: 9px;\n"
"border-bottom-left-radius: 9px;\n"
"}")
        self.epo.setMinimum(1)
        self.epo.setMaximum(20)
        self.epo.setOrientation(QtCore.Qt.Horizontal)
        self.epo.setObjectName("epo")
        self.verticalLayout.addWidget(self.epo)
        self.verticalLayoutWidget_4 = QtWidgets.QWidget(self.frame_4)
        self.verticalLayoutWidget_4.setGeometry(QtCore.QRect(420, 20, 141, 111))
        self.verticalLayoutWidget_4.setObjectName("verticalLayoutWidget_4")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_4)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_5 = QtWidgets.QLabel(self.verticalLayoutWidget_4)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("color: rgb(186, 220, 88)")
        self.label_5.setObjectName("label_5")
        self.verticalLayout_2.addWidget(self.label_5)
        self.label_6 = QtWidgets.QLabel(self.verticalLayoutWidget_4)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_6.setFont(font)
        self.label_6.setStyleSheet("color: rgb(240, 147, 43)")
        self.label_6.setObjectName("label_6")
        self.verticalLayout_2.addWidget(self.label_6)
        self.progressBar = QtWidgets.QProgressBar(self.frame_4)
        self.progressBar.setGeometry(QtCore.QRect(300, 230, 201, 20))
        self.progressBar.setStyleSheet("QProgressBar:horizontal {\n"
"border: 1px solid gray;\n"
"border-radius: 8px;\n"
"background: black;\n"
"padding: 1px;\n"
"}\n"
"QProgressBar::chunk:horizontal {\n"
"background: qlineargradient(x1: 0, y1: 0.5, x2: 1, y2: 0.5, stop: 0 rgb(32, 191, 107), stop: 1 black);\n"
"}")
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.train = QtWidgets.QPushButton(self.frame_4)
        self.train.setGeometry(QtCore.QRect(30, 150, 101, 41))
        self.train.setStyleSheet("background-color: rgb(32, 191, 107);\n"
"color: rgb(32, 191, 107)")
        self.train.setObjectName("train")
        self.label_7 = QtWidgets.QLabel(self.frame_4)
        self.label_7.setGeometry(QtCore.QRect(220, 220, 71, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_7.setFont(font)
        self.label_7.setStyleSheet("color: rgb(32, 191, 107)")
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.frame_4)
        self.label_8.setGeometry(QtCore.QRect(240, 150, 51, 21))
        self.label_8.setStyleSheet("color: rgb(235, 59, 90)")
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.frame_4)
        self.label_9.setGeometry(QtCore.QRect(234, 180, 61, 21))
        self.label_9.setStyleSheet("color: rgb(32, 191, 107)\n"
"")
        self.label_9.setObjectName("label_9")
        self.accuracy = QtWidgets.QLabel(self.frame_4)
        self.accuracy.setGeometry(QtCore.QRect(300, 180, 141, 21))
        self.accuracy.setStyleSheet("color:rgb(32, 191, 107)")
        self.accuracy.setText("")
        self.accuracy.setObjectName("accuracy")
        self.loss = QtWidgets.QLabel(self.frame_4)
        self.loss.setGeometry(QtCore.QRect(300, 150, 151, 21))
        self.loss.setStyleSheet("color: rgb(235, 59, 90)")
        self.loss.setText("")
        self.loss.setObjectName("loss")
        self.line = QtWidgets.QFrame(self.frame_4)
        self.line.setGeometry(QtCore.QRect(183, 0, 20, 291))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.next3 = QtWidgets.QPushButton(self.frame_4)
        self.next3.setGeometry(QtCore.QRect(510, 230, 81, 41))
        self.next3.setStyleSheet("color: rgb(15, 185, 177);\n"
"background-color: rgb(15, 185, 177)")
        self.next3.setIcon(icon5)
        self.next3.setIconSize(QtCore.QSize(20, 20))
        self.next3.setObjectName("next3")
        self.stackedWidget.addWidget(self.page3)
        self.page4 = QtWidgets.QWidget()
        self.page4.setObjectName("page4")
        self.frame_6 = QtWidgets.QFrame(self.page4)
        self.frame_6.setGeometry(QtCore.QRect(0, 0, 601, 291))
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.frame_7 = QtWidgets.QFrame(self.frame_6)
        self.frame_7.setGeometry(QtCore.QRect(10, 0, 601, 291))
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.test = QtWidgets.QPushButton(self.frame_7)
        self.test.setGeometry(QtCore.QRect(20, 50, 101, 51))
        self.test.setStyleSheet("background-color: rgb(15, 185, 177);\n"
"color: rgb(15, 185, 177)")
        self.test.setObjectName("test")
        self.feedback = QtWidgets.QPlainTextEdit(self.frame_7)
        self.feedback.setEnabled(False)
        self.feedback.setGeometry(QtCore.QRect(170, 0, 191, 291))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.feedback.setFont(font)
        self.feedback.setStyleSheet("color:rgb(235, 59, 90)\n"
"")
        self.feedback.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.feedback.setFrameShadow(QtWidgets.QFrame.Raised)
        self.feedback.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.feedback.setPlainText("")
        self.feedback.setBackgroundVisible(False)
        self.feedback.setObjectName("feedback")
        self.line_3 = QtWidgets.QFrame(self.frame_7)
        self.line_3.setWindowModality(QtCore.Qt.NonModal)
        self.line_3.setGeometry(QtCore.QRect(150, 0, 20, 291))
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.line_4 = QtWidgets.QFrame(self.frame_7)
        self.line_4.setGeometry(QtCore.QRect(360, 0, 20, 291))
        self.line_4.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.feedback_3 = QtWidgets.QPlainTextEdit(self.frame_7)
        self.feedback_3.setEnabled(False)
        self.feedback_3.setGeometry(QtCore.QRect(380, 0, 211, 211))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.feedback_3.setFont(font)
        self.feedback_3.setStyleSheet("color:rgb(15, 185, 177)\n"
"")
        self.feedback_3.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.feedback_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.feedback_3.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.feedback_3.setBackgroundVisible(False)
        self.feedback_3.setObjectName("feedback_3")
        self.next4 = QtWidgets.QPushButton(self.frame_7)
        self.next4.setGeometry(QtCore.QRect(500, 240, 81, 41))
        self.next4.setStyleSheet("color: rgb(15, 185, 177);\n"
"background-color: rgb(15, 185, 177)")
        self.next4.setIcon(icon5)
        self.next4.setIconSize(QtCore.QSize(20, 20))
        self.next4.setObjectName("next4")
        self.verticalLayoutWidget_5 = QtWidgets.QWidget(self.frame_7)
        self.verticalLayoutWidget_5.setGeometry(QtCore.QRect(0, 110, 141, 171))
        self.verticalLayoutWidget_5.setObjectName("verticalLayoutWidget_5")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_5)
        self.verticalLayout_5.setContentsMargins(1, 1, 1, 1)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.stop_4 = QtWidgets.QRadioButton(self.verticalLayoutWidget_5)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(232, 65, 24))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(232, 65, 24))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(232, 65, 24))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(232, 65, 24))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(232, 65, 24))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(232, 65, 24))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(232, 65, 24))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(232, 65, 24))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(232, 65, 24))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.stop_4.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Microsoft New Tai Lue")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.stop_4.setFont(font)
        self.stop_4.setStyleSheet("color: rgb(232, 65, 24);\n"
"background-color:transparent")
        self.stop_4.setIcon(icon2)
        self.stop_4.setIconSize(QtCore.QSize(20, 20))
        self.stop_4.setObjectName("stop_4")
        self.verticalLayout_5.addWidget(self.stop_4)
        self.pause_4 = QtWidgets.QRadioButton(self.verticalLayoutWidget_5)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 151, 230))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 151, 230))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 151, 230))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 151, 230))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 151, 230))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 151, 230))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 151, 230))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 151, 230))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 151, 230))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.pause_4.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Microsoft New Tai Lue")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.pause_4.setFont(font)
        self.pause_4.setStyleSheet("color: rgb(0, 151, 230)\n"
"")
        self.pause_4.setIcon(icon3)
        self.pause_4.setIconSize(QtCore.QSize(20, 20))
        self.pause_4.setObjectName("pause_4")
        self.verticalLayout_5.addWidget(self.pause_4)
        self.continue_5 = QtWidgets.QRadioButton(self.verticalLayoutWidget_5)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(76, 209, 55))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(76, 209, 55))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(76, 209, 55))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(76, 209, 55))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(76, 209, 55))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(76, 209, 55))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(76, 209, 55))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(76, 209, 55))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(76, 209, 55))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 54, 64))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.continue_5.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Microsoft New Tai Lue")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.continue_5.setFont(font)
        self.continue_5.setStyleSheet("color: rgb(76, 209, 55)\n"
"")
        self.continue_5.setIcon(icon4)
        self.continue_5.setIconSize(QtCore.QSize(20, 20))
        self.continue_5.setChecked(True)
        self.continue_5.setObjectName("continue_5")
        self.verticalLayout_5.addWidget(self.continue_5)
        self.stackedWidget.addWidget(self.page4)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMainWindow.menuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 601, 21))
        self.menubar.setStyleSheet("QMenuBar::item {\n"
"    background-color: transparent;\n"
"    color : rgb(245, 246, 250)\n"
"}\n"
"QTextEdit{\n"
"    border: 0\n"
"}")
        self.menubar.setDefaultUp(False)
        self.menubar.setObjectName("menubar")
        self.menuClose = QtWidgets.QMenu(self.menubar)
        self.menuClose.setStyleSheet("color:rgb(43, 203, 186)")
        self.menuClose.setObjectName("menuClose")
        self.menuDelete = QtWidgets.QMenu(self.menubar)
        self.menuDelete.setStyleSheet("color:rgb(235, 59, 90)")
        self.menuDelete.setObjectName("menuDelete")
        MainWindow.setMenuBar(self.menubar)
        self.close = QtWidgets.QAction(MainWindow)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap("images/processing-18-694763.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.close.setIcon(icon10)
        self.close.setObjectName("close")
        self.actionTrain_data = QtWidgets.QAction(MainWindow)
        icon11 = QtGui.QIcon()
        icon11.addPixmap(QtGui.QPixmap("images/47-512.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionTrain_data.setIcon(icon11)
        self.actionTrain_data.setObjectName("actionTrain_data")
        self.actionProcessed_data = QtWidgets.QAction(MainWindow)
        self.actionProcessed_data.setIcon(icon11)
        self.actionProcessed_data.setObjectName("actionProcessed_data")
        self.actionModel = QtWidgets.QAction(MainWindow)
        self.actionModel.setIcon(icon11)
        self.actionModel.setObjectName("actionModel")
        self.menuClose.addAction(self.close)
        self.menuDelete.addAction(self.actionTrain_data)
        self.menuDelete.addAction(self.actionProcessed_data)
        self.menuDelete.addAction(self.actionModel)
        self.menubar.addAction(self.menuClose.menuAction())
        self.menubar.addAction(self.menuDelete.menuAction())
        self.next1.clicked.connect(self.procc_page)
        self.next2.clicked.connect(self.train_page)
        self.next3.clicked.connect(self.test_page)
        self.next4.clicked.connect(self.rec_page)
        self.run.clicked.connect(self.record)
        self.process.clicked.connect(self.processed)
        self.sample.clicked.connect(self.samp)
        self.train.clicked.connect(self.tr)
        self.test.clicked.connect(self.tes)
        
            

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    
    def exit():
        sys.exit()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "AI-Tool"))
        self.label.setText(_translate("MainWindow", "Sec"))
        self.run.setText(_translate("MainWindow", "Start"))
        self.stop_3.setText(_translate("MainWindow", "Stop"))
        self.pause_3.setText(_translate("MainWindow", "Pause"))
        self.continue_4.setText(_translate("MainWindow", "Continue"))
        self.label_2.setText(_translate("MainWindow", "Batch size"))
        self.next1.setText(_translate("MainWindow", "Next"))
        self.process.setText(_translate("MainWindow", "Data process"))
        self.label_3.setText(_translate("MainWindow", "status"))
        self.label_4.setText(_translate("MainWindow", "final length"))
        self.sample.setText(_translate("MainWindow", "sample image"))
        self.next2.setText(_translate("MainWindow", "Next"))
        self.alexnetwork.setText(_translate("MainWindow", "Alex Net"))
        self.googlenetwork.setText(_translate("MainWindow", "Google Net"))
        self.label_5.setText(_translate("MainWindow", "Learning_rate"))
        self.label_6.setText(_translate("MainWindow", "Epoch"))
        self.train.setText(_translate("MainWindow", "Train"))
        self.label_7.setText(_translate("MainWindow", "Training :"))
        self.label_8.setText(_translate("MainWindow", "Loss :"))
        self.label_9.setText(_translate("MainWindow", "Accuracy :"))
        self.next3.setText(_translate("MainWindow", "Next"))
        self.test.setText(_translate("MainWindow", "Deploy"))
        self.feedback_3.setPlainText(_translate("MainWindow", "\n"
"\n"
"\n"
"\n"
"               © 2019 \n"
"     Jack Technolog.Pvt.Ltd.\n"
"          All rights reserved.\n"
"\n"
"\n"
""))
        self.next4.setText(_translate("MainWindow", "Next"))
        self.stop_4.setText(_translate("MainWindow", "Stop"))
        self.pause_4.setText(_translate("MainWindow", "Pause"))
        self.continue_5.setText(_translate("MainWindow", "Continue"))
        self.menuClose.setTitle(_translate("MainWindow", "Operation"))
        self.menuDelete.setTitle(_translate("MainWindow", "Delete"))
        self.close.setText(_translate("MainWindow", "close"))
        self.actionTrain_data.setText(_translate("MainWindow", "train data"))
        self.actionProcessed_data.setText(_translate("MainWindow", "processed data"))
        self.actionModel.setText(_translate("MainWindow", "model"))

import resourse_rc


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
