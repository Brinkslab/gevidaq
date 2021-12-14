# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 18:34:44 2021

@author: xinmeng
"""
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QWidget,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QGroupBox,
    QTabWidget,
    QGraphicsView,
    QGraphicsScene,
    QListWidget,
    QSizePolicy,
    QLabel,
    QComboBox,
    QLayout,
    QStackedWidget,
    QSpinBox,
    QLineEdit,
)

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont
import sys
import os

# Ensure that the Widget can be run either independently or as part of Tupolev.
if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname + "/../")
from CoordinatesManager import DMDActuator, Registrator, CoordinateTransformations
from ImageAnalysis.ImageProcessing import ProcessImage
from StylishQT import MySwitch, roundQGroupBox, SquareImageView
from GeneralUsage.ThreadingFunc import run_in_thread

import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np
import StylishQT


class DMDminiWidgetUI(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #        os.chdir('./')# Set directory to current folder.
        self.setFont(QFont("Arial"))

        self.setMinimumHeight(150)
        self.setWindowTitle("DMD mini")
        self.layout = QGridLayout(self)

        # Set the timing between dark phase, in micro-second.
        self.gap_between_dark_phase = 9000000

        DMDminiWidgetContainer = StylishQT.roundQGroupBox(
            title="DMD-mini", background_color="azure"
        )
        DMDminiWidgetContainer.setFixedHeight(150)
        DMDminiWidgetContainer.setFixedWidth(80)
        DMDminiWidgetContainerLayout = QGridLayout()

        self.connect_button = StylishQT.connectButton()
        self.connect_button.setFixedWidth(60)
        self.connect_button.clicked.connect(self.connectDMD)

        self.project_button = QPushButton("Fully proj.")
        self.project_button.setFixedWidth(60)
        self.project_button.setStyleSheet("QPushButton {background-color: #99FFCC;}")
        self.project_button.setCheckable(True)
        self.project_button.clicked.connect(
            lambda: run_in_thread(self.project_full_white)
        )

        DMDminiWidgetContainerLayout.addWidget(self.connect_button, 0, 0)
        DMDminiWidgetContainerLayout.addWidget(self.project_button, 1, 0)

        DMDminiWidgetContainer.setLayout(DMDminiWidgetContainerLayout)

        self.layout.addWidget(DMDminiWidgetContainer, 0, 0)

    def connectDMD(self):
        if self.connect_button.isChecked():
            # self.connect_button.setEnabled(False)
            self.DMD_actuator = DMDActuator.DMDActuator()

        else:
            self.connect_button.setChecked(False)
            self.DMD_actuator.disconnect_DMD()
            del self.DMD_actuator

    def project_full_white(self):

        if self.project_button.isChecked():
            self.project_button.setText("Stop projecting")
            self.DMD_actuator.send_data_to_DMD(np.ones((1024, 768)))

            repeat = True
            # frame_time = int(self.frame_rate_textbox.text())
            self.DMD_actuator.set_repeat(repeat)
            self.DMD_actuator.set_timing(self.gap_between_dark_phase)

            # Set the binary mode of DMD.
            ALP_BIN_MODE = 2104  # 	Binary mode: select from ALP_BIN_NORMAL and ALP_BIN_UNINTERRUPTED (AlpSeqControl)

            ALP_BIN_NORMAL = 2105  # 	Normal operation with progammable dark phase
            ALP_BIN_UNINTERRUPTED = 2106  # 	Operation without dark phase

            self.DMD_actuator.DMD.SeqControl(
                controlType=ALP_BIN_MODE, value=ALP_BIN_UNINTERRUPTED
            )
            print("set to ALP_BIN_UNINTERRUPTED, no frame switching.")

            self.DMD_actuator.start_projection()

        else:
            self.project_button.setText("Fully project")
            self.DMD_actuator.stop_projection()
            self.DMD_actuator.free_memory()


if __name__ == "__main__":

    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        mainwin = DMDminiWidgetUI()
        mainwin.show()
        app.exec_()

    run_app()
