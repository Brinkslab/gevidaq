# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:54:35 2020

@author: xinmeng
"""

from __future__ import division
import sys

sys.path.append("../")
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

import pyqtgraph as pg
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

from ThorlabsFilterSlider.filterpyserial import ELL9Filter


class FilterSliderWidgetUI(QWidget):

    #    waveforms_generated = pyqtSignal(object, object, list, int)
    #    SignalForContourScanning = pyqtSignal(int, int, int, np.ndarray, np.ndarray)
    #    MessageBack = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #        os.chdir('./')# Set directory to current folder.
        self.setFont(QFont("Arial"))

        self.resize(265, 130)
        self.setWindowTitle("FilterSliderWidget")
        self.layout = QGridLayout(self)

        # **************************************************************************************************************************************
        # --------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------GUI for Filter movement----------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------------------------------
        # **************************************************************************************************************************************
        ControlContainer = StylishQT.roundQGroupBox(
            title="Filter Control", background_color="azure"
        )
        ControlContainerLayout = QGridLayout()

        ND_filtercontrolContainer = StylishQT.roundQGroupBox(
            title="2P ND", background_color="azure"
        )
        self.filtercontrolLayout = QGridLayout()
        self.filtercontrolLayout.setSpacing(2)

        self.FilterButtongroup_1 = QButtonGroup(self)

        self.filter1_pos0 = QPushButton("0")
        self.filter1_pos0.setCheckable(True)
        self.FilterButtongroup_1.addButton(self.filter1_pos0)
        self.filtercontrolLayout.addWidget(self.filter1_pos0, 0, 1)
        #        self.filter1_pos0.clicked.connect(lambda: self.filter_move_towards("COM9", 0))

        self.filter1_pos1 = QPushButton("1")
        self.filter1_pos1.setCheckable(True)
        self.FilterButtongroup_1.addButton(self.filter1_pos1)
        self.filtercontrolLayout.addWidget(self.filter1_pos1, 0, 2)
        #        self.filter1_pos1.clicked.connect(lambda: self.filter_move_towards("COM9", 1))

        self.filter1_pos2 = QPushButton("2")
        self.filter1_pos2.setCheckable(True)
        self.FilterButtongroup_1.addButton(self.filter1_pos2)
        self.filtercontrolLayout.addWidget(self.filter1_pos2, 0, 3)
        #        self.filter1_pos2.clicked.connect(lambda: self.filter_move_towards("COM9", 2))

        self.filter1_pos3 = QPushButton("3")
        self.filter1_pos3.setCheckable(True)
        self.FilterButtongroup_1.addButton(self.filter1_pos3)
        self.filtercontrolLayout.addWidget(self.filter1_pos3, 0, 4)
        #        self.filter1_pos3.clicked.connect(lambda: self.filter_move_towards("COM9", 3))
        self.FilterButtongroup_1.setExclusive(True)
        self.FilterButtongroup_1.buttonClicked[int].connect(self.DecodeFilterMove)

        self.FilterButtongroup_2 = QButtonGroup(self)

        self.filter2_pos0 = QPushButton("0")
        self.filter2_pos0.setCheckable(True)
        self.FilterButtongroup_2.addButton(self.filter2_pos0)
        self.filtercontrolLayout.addWidget(self.filter2_pos0, 1, 1)
        #        self.filter1_pos0.clicked.connect(lambda: self.filter_move_towards("COM9", 0))

        self.filter2_pos1 = QPushButton("0.1")
        self.filter2_pos1.setCheckable(True)
        self.FilterButtongroup_2.addButton(self.filter2_pos1)
        self.filtercontrolLayout.addWidget(self.filter2_pos1, 1, 2)
        #        self.filter1_pos1.clicked.connect(lambda: self.filter_move_towards("COM9", 1))

        self.filter2_pos2 = QPushButton("0.3")
        self.filter2_pos2.setCheckable(True)
        self.FilterButtongroup_2.addButton(self.filter2_pos2)
        self.filtercontrolLayout.addWidget(self.filter2_pos2, 1, 3)
        #        self.filter1_pos2.clicked.connect(lambda: self.filter_move_towards("COM9", 2))

        self.filter2_pos3 = QPushButton("0.5")
        self.filter2_pos3.setCheckable(True)
        self.FilterButtongroup_2.addButton(self.filter2_pos3)
        self.filtercontrolLayout.addWidget(self.filter2_pos3, 1, 4)
        #        self.filter1_pos3.clicked.connect(lambda: self.filter_move_towards("COM9", 3))
        self.FilterButtongroup_2.setExclusive(True)
        self.FilterButtongroup_2.buttonClicked[int].connect(self.DecodeFilterMove)
        #
        #        self.filtercontrolLayout.addWidget(QLabel('Filter-1 pos: '), 0, 0)
        #
        #        self.filtercontrolLayout.addWidget(QLabel('Filter-2 pos: '), 1, 0)
        #        bGBackupFromIntExt_1 = QButtonGroup(self)
        #
        #        self.filter2_pos0 = QPushButton('0')
        #        self.filter2_pos0.setCheckable(True)
        #        bGBackupFromIntExt_1.addButton(self.filter2_pos0)
        #        self.filtercontrolLayout.addWidget(self.filter2_pos0, 1, 1)
        #        self.filter2_pos0.clicked.connect(lambda: self.filter_move_towards("COM7", 0))
        #
        #        self.filter2_pos1 = QPushButton('0.1')
        #        self.filter2_pos1.setCheckable(True)
        #        bGBackupFromIntExt_1.addButton(self.filter2_pos1)
        #        self.filtercontrolLayout.addWidget(self.filter2_pos1, 1, 2)
        #        self.filter2_pos1.clicked.connect(lambda: self.filter_move_towards("COM7", 1))
        #
        #        self.filter2_pos2 = QPushButton('0.3')
        #        self.filter2_pos2.setCheckable(True)
        #        bGBackupFromIntExt_1.addButton(self.filter2_pos2)
        #        self.filtercontrolLayout.addWidget(self.filter2_pos2, 1, 3)
        #        self.filter2_pos2.clicked.connect(lambda: self.filter_move_towards("COM7", 2))
        #
        #        self.filter2_pos3 = QPushButton('0.5')
        #        self.filter2_pos3.setCheckable(True)
        #        bGBackupFromIntExt_1.addButton(self.filter2_pos3)
        #        self.filtercontrolLayout.addWidget(self.filter2_pos3, 1, 4)
        #        self.filter2_pos3.clicked.connect(lambda: self.filter_move_towards("COM7", 3))

        # ----------------------------------------------------------------------
        #        self.filter1_pos0 =  QDial()
        #        self.filter1_pos0.setMinimum(0)
        #        self.filter1_pos0.setMaximum(3)
        #        self.filter1_pos0.setValue(0)
        #        self.filter1_pos0.setNotchesVisible(True)
        #        self.filter1_pos0.valueChanged.connect(lambda: self.filter_move_towards("COM9", self.filter1_pos0.value()))

        #        self.filter2_pos0 =  QDial()
        #        self.filter2_pos0.setMinimum(0)
        #        self.filter2_pos0.setMaximum(3)
        #        self.filter2_pos0.setValue(0)
        #        self.filter2_pos0.setNotchesVisible(True)
        #        self.filter2_pos0.valueChanged.connect(lambda: self.filter_move_towards("COM7", self.filter2_pos0.value()))
        #
        #        self.filter3_pos0 =  QDial()
        #        self.filter3_pos0.setMinimum(0)
        #        self.filter3_pos0.setMaximum(3)
        #        self.filter3_pos0.setValue(0)
        #        self.filter3_pos0.setNotchesVisible(True)
        #        self.filter3_pos0.valueChanged.connect(lambda: self.filter_move_towards("COM15", self.filter3_pos0.value()))

        # ----------------------------------------------------------------------
        #        self.filter1_pos0 = QSlider(Qt.Horizontal)
        #        self.filter1_pos0.setMinimum(0)
        #        self.filter1_pos0.setMaximum(3)
        #        self.filter1_pos0.setTickPosition(QSlider.TicksBothSides)
        #        self.filter1_pos0.setTickInterval(1)
        #        self.filter1_pos0.setSingleStep(1)
        #        self.filter1_pos0.sliderReleased.connect(lambda: self.filter_move_towards("COM9", self.filter1_pos0.value()))
        #
        #        self.filter2_pos0 = QSlider(Qt.Horizontal)
        #        self.filter2_pos0.setMinimum(0)
        #        self.filter2_pos0.setMaximum(3)
        #        self.filter2_pos0.setTickPosition(QSlider.TicksBothSides)
        #        self.filter2_pos0.setTickInterval(1)
        #        self.filter2_pos0.setSingleStep(1)
        #        self.filter2_pos0.sliderReleased.connect(lambda: self.filter_move_towards("COM7", self.filter2_pos0.value()))
        #
        #        self.filter3_pos0 = QSlider(Qt.Vertical)
        #        self.filter3_pos0.setMinimum(0)
        #        self.filter3_pos0.setMaximum(1)
        #        self.filter3_pos0.setTickPosition(QSlider.TicksBothSides)
        #        self.filter3_pos0.setTickInterval(1)
        #        self.filter3_pos0.setSingleStep(1)
        #        self.filter3_pos0.sliderReleased.connect(lambda: self.filter_move_towards("COM15", self.filter3_pos0.value()))

        #        self.filtercontrolLayout.addWidget(QLabel('ND 0 | 1 | 2 | 3'), 0, 1)
        #        self.filtercontrolLayout.addWidget(self.filter1_pos0, 1, 1)
        #        self.filtercontrolLayout.addWidget(QLabel('ND 0 | 0.1 | 0.3 | 0.5'), 0, 2)
        #        self.filtercontrolLayout.addWidget(self.filter2_pos0, 1, 2)

        #        self.filtercontrolLayout.addWidget(self.filter3_pos0, 1, 3)

        #        oImage = QImage('./Icons/filtersliderpanel.png')
        ##        sImage = oImage.scaled(QSize(292,208))                   # resize Image to widgets size
        #        palette = QPalette()
        #        palette.setBrush(QPalette.Window, QBrush(oImage))

        EM_filtercontrolContainer = StylishQT.roundQGroupBox(
            title="Emission", background_color="honeydew"
        )
        self.EM_filtercontrolContainerLayout = QGridLayout()
        self.EM_filtercontrolContainerLayout.setSpacing(2)

        self.FilterButtongroup_3 = QButtonGroup(self)

        self.filter3_pos0 = QPushButton("Arch")
        self.filter3_pos0.setCheckable(True)
        self.FilterButtongroup_3.addButton(self.filter3_pos0)
        self.EM_filtercontrolContainerLayout.addWidget(self.filter3_pos0, 1, 0)
        #        self.filter1_pos0.clicked.connect(lambda: self.filter_move_towards("COM9", 0))

        self.filter3_pos1 = QPushButton("Citrine")
        self.filter3_pos1.setCheckable(True)
        self.FilterButtongroup_3.addButton(self.filter3_pos1)
        self.EM_filtercontrolContainerLayout.addWidget(self.filter3_pos1, 0, 0)
        self.FilterButtongroup_3.setExclusive(True)
        self.FilterButtongroup_3.buttonClicked[int].connect(self.DecodeFilterMove)

        EM_filtercontrolContainer.setLayout(self.EM_filtercontrolContainerLayout)
        EM_filtercontrolContainer.setFixedWidth(65)

        ND_filtercontrolContainer.setLayout(self.filtercontrolLayout)
        # ND_filtercontrolContainer.setFixedHeight(110)
        ND_filtercontrolContainer.setFixedWidth(200)
        #        self.setPalette(palette)
        #        self.setAutoFillBackground(True)

        ControlContainerLayout.addWidget(ND_filtercontrolContainer, 0, 0)
        ControlContainerLayout.addWidget(EM_filtercontrolContainer, 0, 1)
        ControlContainer.setLayout(ControlContainerLayout)

        self.layout.addWidget(ControlContainer, 0, 0)

        # **************************************************************************************************************************************
        # --------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------Fucs for filter movement---------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------------------------------
        # **************************************************************************************************************************************

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

    def DecodeFilterMove(self):

        if self.FilterButtongroup_1.checkedId() == -2:
            self.run_in_thread(self.filter_move_towards("COM9", 0))
        elif self.FilterButtongroup_1.checkedId() == -3:
            self.run_in_thread(self.filter_move_towards("COM9", 1))
        elif self.FilterButtongroup_1.checkedId() == -4:
            self.run_in_thread(self.filter_move_towards("COM9", 2))
        elif self.FilterButtongroup_1.checkedId() == -5:
            self.run_in_thread(self.filter_move_towards("COM9", 3))

        if self.FilterButtongroup_2.checkedId() == -2:
            self.run_in_thread(self.filter_move_towards("COM7", 0))
        elif self.FilterButtongroup_2.checkedId() == -3:
            self.run_in_thread(self.filter_move_towards("COM7", 1))
        elif self.FilterButtongroup_2.checkedId() == -4:
            self.run_in_thread(self.filter_move_towards("COM7", 2))
        elif self.FilterButtongroup_2.checkedId() == -5:
            self.run_in_thread(self.filter_move_towards("COM7", 3))

        if self.FilterButtongroup_3.checkedId() == -2:
            # Move to Arch
            self.run_in_thread(self.filter_move_towards("COM15", 0))
        elif self.FilterButtongroup_3.checkedId() == -3:
            self.run_in_thread(self.filter_move_towards("COM15", 1))

    # def start_up_event(self):
    #     ports = ["COM9", "COM7", "COM15"]
    #     try:
    #         for port in ports:
    #             ELL9Filter_ins = ELL9Filter(port)
    #             pos =

    def filter_move_towards(self, COMport, pos):
        ELL9Filter_ins = ELL9Filter(COMport)
        ELL9Filter_ins.moveToPosition(pos)

    def update_slider_current_pos(self, current_pos):
        #        .setValue(current_pos)
        print("Slider current position: {}".format(current_pos))


if __name__ == "__main__":

    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        pg.setConfigOptions(imageAxisOrder="row-major")
        mainwin = FilterSliderWidgetUI()
        mainwin.show()
        app.exec_()

    run_app()
