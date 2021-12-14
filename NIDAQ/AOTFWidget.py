# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 12:18:12 2020

@author: xinmeng
"""

from __future__ import division
import sys

sys.path.append("../")
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPoint, QRect, QObject
from PyQt5.QtGui import (
    QColor,
    QPen,
    QPixmap,
    QIcon,
    QTextCursor,
    QFont,
    QPainter,
    QBrush,
)

from PyQt5.QtWidgets import (
    QWidget,
    QButtonGroup,
    QLabel,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QGridLayout,
    QPushButton,
    QGroupBox,
    QLineEdit,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QMessageBox,
    QTabWidget,
    QCheckBox,
    QRadioButton,
    QFileDialog,
    QProgressBar,
    QTextEdit,
    QStackedLayout,
)

import pyqtgraph as pg
import time
import threading
import os

# Append parent folder to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import StylishQT
from NIDAQ.ServoMotor import Servo
from NIDAQ.DAQoperator import DAQmission


class AOTFWidgetUI(QWidget):

    #    waveforms_generated = pyqtSignal(object, object, list, int)
    #    SignalForContourScanning = pyqtSignal(int, int, int, np.ndarray, np.ndarray)
    #    MessageBack = pyqtSignal(str)
    sig_lasers_status_changed = pyqtSignal(dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #        os.chdir('./')# Set directory to current folder.
        self.setFont(QFont("Arial"))

        #        self.setMinimumSize(1350,900)
        self.setWindowTitle("StageWidget")
        self.layout = QGridLayout(self)

        # **************************************************************************************************************************************
        # --------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------GUI for AOTF---------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------------------------------
        # **************************************************************************************************************************************

        AOTFcontrolContainer = StylishQT.roundQGroupBox(
            title="AOTF control", background_color="azure"
        )

        self.AOTFstackedLayout = QStackedLayout()

        # self.AOTFdisabledWidget = QWidget()
        self.AOTFdisabledWidget = QLabel(
            "AOTF not available due to running registration procedure"
        )
        self.AOTFdisabledWidget.setWordWrap(True)

        self.AOTFcontrolWidget = QWidget()
        self.AOTFcontrolLayout = QGridLayout()
        self.AOTFcontrolWidget.setLayout(self.AOTFcontrolLayout)

        self.shutter640Button = StylishQT.checkableButton(
            Icon_path="./Icons/shutter.png", background_color="#DEC8C4"
        )

        self.slider640 = QSlider(Qt.Horizontal)
        self.slider640.setMinimum(0)
        self.slider640.setMaximum(500)
        self.slider640.setTickPosition(QSlider.TicksBothSides)
        self.slider640.setTickInterval(100)
        self.slider640.setSingleStep(1)
        self.line640 = QLineEdit(self)
        self.line640.setFixedWidth(46)
        self.slider640.sliderReleased.connect(lambda: self.updatelinevalue(640))
        self.slider640.sliderReleased.connect(
            lambda: self.execute_tread_single_sample_analog("640AO")
        )
        self.line640.returnPressed.connect(lambda: self.updateslider(640))

        self.switchbutton_blankingAll = StylishQT.MySwitch(
            "Blanking ON", "spring green", "Blanking OFF", "indian red", width=60
        )
        self.switchbutton_blankingAll.clicked.connect(
            lambda: self.execute_tread_single_sample_digital("640blanking")
        )
        self.AOTFcontrolLayout.addWidget(self.switchbutton_blankingAll, 0, 2, 1, 2)

        self.shutter532Button = StylishQT.checkableButton(
            Icon_path="./Icons/shutter.png", background_color="#CDDEC4"
        )

        self.slider532 = QSlider(Qt.Horizontal)
        self.slider532.setMinimum(0)
        self.slider532.setMaximum(500)
        self.slider532.setTickPosition(QSlider.TicksBothSides)
        self.slider532.setTickInterval(100)
        self.slider532.setSingleStep(1)
        self.line532 = QLineEdit(self)
        self.line532.setFixedWidth(46)
        self.slider532.sliderReleased.connect(lambda: self.updatelinevalue(532))
        self.slider532.sliderReleased.connect(
            lambda: self.execute_tread_single_sample_analog("532AO")
        )
        self.line532.returnPressed.connect(lambda: self.updatesider(532))

        self.switchbutton_532 = StylishQT.MySwitch(
            "ON", "green", "OFF", "lime green", width=32
        )
        self.switchbutton_532.clicked.connect(
            lambda: self.execute_tread_single_sample_digital("532blanking")
        )
        # self.AOTFcontrolLayout.addWidget(self.switchbutton_532, 1, 1)

        self.slider488 = QSlider(Qt.Horizontal)
        self.slider488.setMinimum(0)
        self.slider488.setMaximum(500)
        self.slider488.setTickPosition(QSlider.TicksBothSides)
        self.slider488.setTickInterval(100)
        self.slider488.setSingleStep(1)
        self.line488 = QLineEdit(self)
        self.line488.setFixedWidth(46)
        self.slider488.sliderReleased.connect(lambda: self.updatelinevalue(488))
        self.slider488.sliderReleased.connect(
            lambda: self.execute_tread_single_sample_analog("488AO")
        )
        self.line488.returnPressed.connect(lambda: self.updatesider(488))

        self.switchbutton_488 = StylishQT.MySwitch(
            "ON", "blue", "OFF", "corn flower blue", width=32
        )
        self.switchbutton_488.clicked.connect(
            lambda: self.execute_tread_single_sample_digital("488blanking")
        )
        # self.AOTFcontrolLayout.addWidget(self.switchbutton_488, 2, 1)

        self.shutter488Button = StylishQT.checkableButton(
            Icon_path="./Icons/shutter.png", background_color="#C4DDDE"
        )
        self.shutter488Button.clicked.connect(lambda: self.shutter_CW_action("488"))

        self.AOTFcontrolLayout.addWidget(self.shutter640Button, 1, 4)
        self.AOTFcontrolLayout.addWidget(self.slider640, 1, 2)
        self.AOTFcontrolLayout.addWidget(self.line640, 1, 3)

        self.AOTFcontrolLayout.addWidget(self.shutter532Button, 2, 4)
        self.AOTFcontrolLayout.addWidget(self.slider532, 2, 2)
        self.AOTFcontrolLayout.addWidget(self.line532, 2, 3)

        self.AOTFcontrolLayout.addWidget(self.shutter488Button, 3, 4)
        self.AOTFcontrolLayout.addWidget(self.slider488, 3, 2)
        self.AOTFcontrolLayout.addWidget(self.line488, 3, 3)

        self.AOTFstackedLayout.addWidget(self.AOTFcontrolWidget)
        self.AOTFstackedLayout.addWidget(self.AOTFdisabledWidget)
        self.AOTFstackedLayout.setCurrentIndex(0)

        AOTFcontrolContainer.setLayout(self.AOTFstackedLayout)
        AOTFcontrolContainer.setMaximumHeight(170)
        self.layout.addWidget(AOTFcontrolContainer, 1, 0)

        self.lasers_status = {}
        self.lasers_status["488"] = [False, 0]
        self.lasers_status["532"] = [False, 0]
        self.lasers_status["640"] = [False, 0]

        thread = threading.Thread(target=self.start_up_event)
        thread.start()
        self.shutter488Button.setChecked(False)
        # **************************************************************************************************************************************
        # --------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------Fuc for AOTF---------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------------------------------
        # **************************************************************************************************************************************

    def start_up_event(self):
        try:
            print("Servo position initialization turned off.")
            # servo= Servo()
            # # close the blue shutter
            # servo.rotate(target_servo = 'servo_modulation_1', degree = 0)
        except:
            print("Fail to initialize servo position.")

    def updatelinevalue(self, wavelength):
        if wavelength == 640:
            self.line640.setText(str(self.slider640.value() / 100))
        if wavelength == 532:
            self.line532.setText(str(self.slider532.value() / 100))
        if wavelength == 488:
            self.line488.setText(str(self.slider488.value() / 100))

    def updateslider(self, wavelength):
        # self.slider640.setSliderPosition(int(float((self.line640.text())*100)))
        if wavelength == 640:
            self.slider640.setValue(int(float(self.line640.text()) * 100))
        if wavelength == 532:
            self.slider532.setValue(int(float(self.line532.text()) * 100))
        if wavelength == 488:
            self.slider488.setValue(int(float(self.line488.text()) * 100))

    def execute_tread_single_sample_analog(self, channel):
        daq = DAQmission()
        if channel == "640AO":
            self.lasers_status["640"][1] = self.slider640.value()

            daq.sendSingleAnalog(channel, self.slider640.value() / 100)

        elif channel == "532AO":
            self.lasers_status["532"][1] = self.slider532.value()
            daq.sendSingleAnalog(channel, self.slider532.value() / 100)

        elif channel == "488AO":
            self.lasers_status["488"][1] = self.slider488.value()
            daq.sendSingleAnalog(channel, self.slider488.value() / 100)

        self.sig_lasers_status_changed.emit(self.lasers_status)

    def execute_tread_single_sample_digital(self, channel):
        daq = DAQmission()
        if channel == "640blanking":
            if self.switchbutton_blankingAll.isChecked():
                self.lasers_status["640"][0] = True
                daq.sendSingleDigital(channel, True)

            else:
                self.lasers_status["640"][0] = False
                daq.sendSingleDigital(channel, False)

        elif channel == "532blanking":
            if self.switchbutton_532.isChecked():
                self.lasers_status["532"][0] = True
                daq.sendSingleDigital(channel, True)

            else:
                self.lasers_status["532"][0] = False
                daq.sendSingleDigital(channel, False)

        elif channel == "488blanking":
            if self.switchbutton_488.isChecked():
                self.lasers_status["488"][0] = True
                daq.sendSingleDigital(channel, True)
            else:
                self.lasers_status["488"][0] = False
                daq.sendSingleDigital(channel, False)

        self.sig_lasers_status_changed.emit(self.lasers_status)

    def shutter_CW_action(self, laser):
        if laser == "488":
            if self.shutter488Button.isChecked():
                servo = Servo()
                servo.rotate(target_servo="servo_modulation_1", degree=180)
            else:
                servo = Servo()
                servo.rotate(target_servo="servo_modulation_1", degree=0)

    def set_registration_mode(self, flag_registration_mode):
        if flag_registration_mode:
            self.AOTFstackedLayout.setCurrentIndex(1)
        else:
            self.AOTFstackedLayout.setCurrentIndex(0)

    def control_for_registration(self, wavelength, value):
        value = int(value)
        daq = DAQmission()

        if value == 0:
            switch = False
        else:
            switch = True

        if wavelength == "640":
            print(wavelength + ":" + str(value))
            print(str(switch))
            daq.sendSingleAnalog("640AO", value)

            daq.sendSingleDigital("640blanking", switch)

        elif wavelength == "532":
            print(wavelength + ":" + str(value))
            print(str(switch))
            daq.sendSingleAnalog("532AO", value)

            daq.sendSingleDigital("640blanking", switch)

        else:
            print(wavelength + ":" + str(value))
            print(str(switch))
            daq.sendSingleAnalog("488AO", value)

            daq.sendSingleDigital("640blanking", switch)


if __name__ == "__main__":

    #    import sys
    sys.path.append("../")

    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        pg.setConfigOptions(imageAxisOrder="row-major")
        mainwin = AOTFWidgetUI()
        mainwin.show()
        app.exec_()

    run_app()
