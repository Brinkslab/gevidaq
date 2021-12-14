# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:42:01 2020

@author: xinmeng
"""

from __future__ import division
import sys
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPoint, QRect, QObject, QSize
from PyQt5.QtGui import QImage, QPalette, QBrush, QFont

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
    QDial,
)

import time
import threading
import sys
import os

# Ensure that the Widget can be run either independently or as part of Tupolev.
if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname + "/../")
import StylishQT

from ThorlabsKCube.KCube_backend import KCube


class KCubeWidgetUI(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setFont(QFont("Arial"))

        self.setWindowTitle("KCube Widget")
        self.layout = QGridLayout(self)

        # The default camera position is the same as Home position.
        self.PMT_pos = 32  # mm

        self.KCube_instance = KCube()
        # -------------------------------GUI------------------------------------
        KCubeContainer = StylishQT.roundQGroupBox(
            title="KCube Widget", background_color="#F8F8FF"
        )
        KCubeContainerLayout = QGridLayout()

        self.connectButton = StylishQT.connectButton()
        self.connectButton.setFixedWidth(70)
        KCubeContainerLayout.addWidget(self.connectButton, 0, 0)
        self.connectButton.clicked.connect(lambda: self.buildCommunication())

        self.PosSwitchButton = StylishQT.MySwitch(
            "PMT pos.", "lemon chiffon", "Camera pos.", "lavender", width=70
        )
        self.PosSwitchButton.setChecked(False)
        self.PosSwitchButton.clicked.connect(self.PosSwitchEvent)
        KCubeContainerLayout.addWidget(self.PosSwitchButton, 0, 1)

        KCubeContainer.setLayout(KCubeContainerLayout)

        self.layout.addWidget(KCubeContainer, 0, 0)
        self.setFixedHeight(90)

    def run_in_thread(self, fn, *args, **kwargs):
        """
        Send target function to thread.
        Usage: lambda: self.run_in_thread(self.fn)

        Parameters
        ----------
        fn : function
            Target function to put in thread.

        Returns
        -------
        thread : TYPE
            Threading handle.

        """
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()

        return thread

    def buildCommunication(self):
        if self.connectButton.isChecked():
            self.connectButton.setEnabled(False)
            self.run_in_thread(self.ConnectMotor)
        else:
            self.connectButton.setChecked(False)
            self.run_in_thread(self.DisconnectMotor)

    def ConnectMotor(self):
        self.KCube_instance.initialize()
        self.KCube_instance.Home()

        self.connectButton.setEnabled(True)
        self.connectButton.setChecked(True)

    def DisconnectMotor(self):
        self.KCube_instance.Exit()

    def PosSwitchEvent(self):
        if self.PosSwitchButton.isChecked():

            self.run_in_thread(lambda: self.run_in_thread(self.KCube_instance.Move(32)))
        else:

            self.run_in_thread(lambda: self.KCube_instance.Home())


if __name__ == "__main__":

    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        mainwin = KCubeWidgetUI()
        mainwin.show()
        app.exec_()

    run_app()
