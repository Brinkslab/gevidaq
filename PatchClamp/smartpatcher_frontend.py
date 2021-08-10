# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 15:15:38 2021

@author: tvdrb
"""


import os
import sys
import numpy as np
import logging

from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QDoubleSpinBox, QGroupBox, QLabel
import pyqtgraph.exporters
import pyqtgraph as pg


import matplotlib.pyplot as plt
from smartpatcher_backend import SmartPatcher


class PatchClampUI(QWidget):
    def __init__(self):
        super().__init__()
        """
        =======================================================================
        ----------------------------- Start of GUI ----------------------------
        =======================================================================
        """
        """
        # ---------------------- General widget settings ---------------------
        """
        self.setWindowTitle("Automatic Patchclamp")
        
        """
        -------------------------- Hardware container -------------------------
        """
        hardwareContainer = QGroupBox()
        hardwareLayout = QGridLayout()
        
        # Button to (dis)connect camera
        self.connect_camera_button = QPushButton(text="Camera", clicked=self.mockfunction)
        self.connect_camera_button.setCheckable(True)
        
        # Button to (dis)connect objective motor
        self.connect_objectivemotor_button = QPushButton(text="Objective motor", clicked=self.mockfunction)
        self.connect_objectivemotor_button.setCheckable(True)
        
        # Button to (dis)connect micromanipulator
        self.connect_micromanipulator_button = QPushButton(text="Micromanipulator", clicked=self.mockfunction)
        self.connect_micromanipulator_button.setCheckable(True)
        
        # Button to (dis)connect amplifier
        self.connect_amplifier_button = QPushButton(text="Amplifier", clicked=self.mockfunction)
        self.connect_amplifier_button.setCheckable(True)
        
        # Button to (dis)connect pressure controller
        self.connect_pressurecontroller_button = QPushButton(text="Pressure controller", clicked=self.mockfunction)
        self.connect_pressurecontroller_button.setCheckable(True)
        
        # Button to stop all hardware in motion
        self.STOP_button = QPushButton(text="Emergency STOP", clicked=self.mockfunction)
        self.STOP_button.setCheckable(True)
        
        hardwareLayout.addWidget(self.connect_camera_button, 0, 0, 1, 1)
        hardwareLayout.addWidget(self.connect_objectivemotor_button, 1, 0, 1, 1)
        hardwareLayout.addWidget(self.connect_micromanipulator_button, 2, 0, 1, 1)
        hardwareLayout.addWidget(self.connect_amplifier_button, 3, 0, 1, 1)
        hardwareLayout.addWidget(self.connect_pressurecontroller_button, 4, 0, 1, 1)
        hardwareLayout.addWidget(self.STOP_button, 5, 0, 1, 1)
        hardwareContainer.setLayout(hardwareLayout)
        
        """
        ------------------------- Camera view display -------------------------
        """
        liveContainer = QGroupBox()
        liveContainer.setMinimumSize(600, 600)
        liveLayout = QGridLayout()
        
        # Display to project live camera view
        liveWidget = pg.ImageView()
        liveWidget.ui.roiBtn.hide()
        liveWidget.ui.menuBtn.hide()
        liveWidget.ui.histogram.hide()
        self.liveView = liveWidget.getView()
        self.liveImageItem = liveWidget.getImageItem()
        # self.canvas.setAutoDownsample(True)
        
        # Button for pausing camera view
        self.request_pause_button = QPushButton(text="Pause live", clicked=self.request_togglelive)
        self.request_pause_button.setCheckable(True)
        
        liveLayout.addWidget(liveWidget, 0, 0, 1, 1)
        liveLayout.addWidget(self.request_pause_button, 1, 0, 1, 1)
        liveContainer.setLayout(liveLayout)
        
        """
        -------------------------- Sensory output display -------------------------
        """
        sensorContainer = QGroupBox()
        sensorContainer.setMinimumSize(400,600)
        sensorLayout = QGridLayout()
        
        sensorWidget = pg.GraphicsLayoutWidget()
        visuals = sensorWidget.addViewBox(0, 0, 1, 1)
        current = sensorWidget.addPlot(1, 0, 1, 1)
        pressure = sensorWidget.addPlot(2, 0, 1, 1)
        
        # # Display to project current graph
        # currentWidget = pg.ImageView()
        # self.currentgraph = currentWidget.getImageItem()
        # self.currentgraph.setAutoDownsample(True)
        # currentWidget.ui.roiBtn.hide()
        # currentWidget.ui.menuBtn.hide()
        # currentWidget.ui.histogram.hide()
        
        # # Display to project pressure graph
        # pressureWidget = pg.ImageView()
        # self.pressuregraph = pressureWidget.getImageItem()
        # self.pressuregraph.setAutoDownsample(True)
        # pressureWidget.ui.roiBtn.hide()
        # pressureWidget.ui.menuBtn.hide()
        # pressureWidget.ui.histogram.hide()
        
        # sensorLayout.addWidget(currentWidget, 0, 0, 1, 1)
        # sensorLayout.addWidget(pressureWidget, 1, 0, 1, 1)
        sensorLayout.addWidget(sensorWidget)
        sensorContainer.setLayout(sensorLayout)
        
        
        
        
        
        """
        ---------------------- Algorithm control buttons ----------------------
        """
        algorithmContainer = QGroupBox()
        algorithmLayout = QGridLayout()
        
        # Button for hard calibration in XY
        request_hardcalibrationxy_button = QPushButton(text="Calibrate XY", clicked=self.mockfunction)
        
        # Button for hard calibration in XYZ
        request_hardcalibrationxyz_button = QPushButton(text="Calibrate XYZ", clicked=self.mockfunction)
        
        # Button for target selection
        request_selecttarget_button = QPushButton(text="Select target", clicked=self.request_selecttarget)
        
        # Button for confirming selected target
        request_confirmtarget_button = QPushButton(text="Confirm target", clicked=self.request_confirmtarget)
        
        # Button for pipette tip detection in XY
        request_detecttip_button = QPushButton(text="Detect tip", clicked=self.mockfunction)
        
        # Button for pipette tip autofocus
        request_autofocustip = QPushButton(text="Autofocus tip", clicked=self.mockfunction)
        
        # Button for gigaseal formation
        request_gigaseal_button = QPushButton(text="Gigaseal", clicked=self.mockfunction)
        
        # Button for break-in
        request_breakin_button = QPushButton(text="Break-in", clicked=self.mockfunction)
        
        # Button for ZAP
        request_zap_button = QPushButton(text="ZAP", clicked=self.mockfunction)
        
        # Button to set pressure
        self.set_pressure_button = QDoubleSpinBox(self)
        self.set_pressure_button.setMinimum(-200)
        self.set_pressure_button.setMaximum(200)
        self.set_pressure_button.setDecimals(0)
        self.set_pressure_button.setValue(0)
        self.set_pressure_button.setSingleStep(1)
        
        # Button to release pressure instantaneous
        request_releasepressure_button = QPushButton(text="Release pressure", clicked=self.mockfunction)
        
        # Button to send pressure to pressure controller
        request_applypressure_button = QPushButton(text="Apply pressure", clicked=self.mockfunction)
        
        algorithmLayout.addWidget(request_hardcalibrationxy_button, 0, 0, 1, 1)
        algorithmLayout.addWidget(request_hardcalibrationxyz_button, 1, 0, 1, 1)
        algorithmLayout.addWidget(request_selecttarget_button, 0, 1, 1, 1)
        algorithmLayout.addWidget(request_confirmtarget_button, 1, 1, 1, 1)
        algorithmLayout.addWidget(request_detecttip_button, 0, 2, 2, 1)
        algorithmLayout.addWidget(request_autofocustip, 0, 3, 2, 1)
        algorithmLayout.addWidget(request_gigaseal_button, 0, 4, 2, 1)
        algorithmLayout.addWidget(request_breakin_button, 0, 5, 2, 1)
        algorithmLayout.addWidget(request_zap_button, 0, 6, 2, 1)
        algorithmLayout.addWidget(QLabel("Pressure (in mBar):"), 0, 7, 1, 1)
        algorithmLayout.addWidget(self.set_pressure_button, 0, 8, 1, 1)
        algorithmLayout.addWidget(request_releasepressure_button, 1, 7, 1, 1)
        algorithmLayout.addWidget(request_applypressure_button, 1, 8, 1, 1)
        algorithmContainer.setLayout(algorithmLayout)
        
        """
        --------------------------- Adding to master --------------------------
        """
        master = QGridLayout()
        master.addWidget(hardwareContainer, 0, 0, 1, 1)
        master.addWidget(liveContainer, 0, 1, 1, 1)
        master.addWidget(sensorContainer, 0, 2, 1, 1)
        master.addWidget(algorithmContainer, 1, 0, 1, 3)

        self.setLayout(master)
        
        """
        =======================================================================
        ----------------------------- End of GUI ------------------------------
        =======================================================================
        """
        
        """
        =======================================================================
        ------------------------ Start up roi manager -------------------------
        =======================================================================
        """
        self.roimanager = ROIManagerGUI(offset=len(self.liveView.addedItems))
        
        """
        =======================================================================
        -------------------------- Start up backend ---------------------------
        =======================================================================
        """
        self.backend = SmartPatcher()
        
        
        
        
    def mockfunction(self):
        print("Button pushed")
        
        
    def request_togglelive(self):
        if self.request_pause_button.isChecked():
            I = plt.imread("./testimage.tif")
            self.liveImageItem.setImage(I)
        else:
            self.liveImageItem.setImage(image=np.zeros((2048,2048)))
        
        
    def request_selecttarget(self):
        """
        The user drags a circular ROI on top of the target cell. The ROI center
        is the target. We first check if a target already exists, if so, we
        recycle it.
        """
        if not self.roimanager.contains('target'):
            target = pg.CircleROI(pos=(0,0), radius=60, movable=True, pen=QPen(QColor(255,255,0), 0))
            self.roimanager.addROI('target')
            self.liveView.addItem(target)
        else:
            idx = self.roimanager.giveROIindex('target')[-1]
            self.liveView.addedItems[idx].translatable = True
            self.liveView.addedItems[idx].setPen(QPen(QColor(255,255,0), 0))
            self.backend.target_coordinates = np.array([None,None])
        
        
    def request_confirmtarget(self):
        """
        If a target is selected, we save the center coordinates of the ROI in
        the camera field of reference.
        """
        if self.roimanager.contains('target'):
            idx = self.roimanager.giveROIindex('target')[-1]
            x,y = self.liveView.addedItems[idx].state['pos'] + self.liveView.addedItems[idx].state['size'] / 2
            self.liveView.addedItems[idx].translatable = False
            self.liveView.addedItems[idx].setPen(QPen(QColor(193,245,240), 0))
            self.backend.target_coordinates = np.array([x,y])
        
        
    def closeEvent(self, event):
        """ Close event
        This method is called when the GUI is shut down. First we need to stop
        the threads that are still running, then we disconnect all hardware to
        be reused in the main widget, only then we accept the close event.
        and quit the widget.
        """
        event.accept()
        
        # Frees the console by quitting the application entirely
        QtWidgets.QApplication.quit() # remove when part of Tupolev!!
        
        


class ROIManagerGUI:
    def __init__(self, offset):
        self.offset = offset
        self.ROInumber = 0
        self.ROIdictionary = {}
    
    def addROI(self, name):
        if name in self.ROIdictionary:
            self.ROIdictionary[name] = self.ROIdictionary[name] + [self.ROInumber]
            self.ROInumber += 1
        else:
            self.ROIdictionary[name] = [self.ROInumber]
            self.ROInumber += 1
        
    def removeROI(self, name, args=None):
        if args == 'all':
            self.ROInumber -= len(self.ROIdictionary[name])
            del self.ROIdictionary[name]
        elif type(args) == int:
            entries = len(self.ROIdictionary[name])
            if entries > args:
                self.ROInumber -= args
                self.ROIdictionary[name] = self.ROIdictionary[name][:-args]
            elif entries == args:
                self.ROInumber -= entries
                del self.ROIdictionary[name]
            else:
                raise ValueError('list out of range')
        
    def giveROIindex(self, name):
        return [x+self.offset for x in self.ROIdictionary[name][:]]
                
    def contains(self, name):
        if name in self.ROIdictionary:
            return True
        else:
            return False





if __name__ == "__main__":
    
    def start_logger():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                # logging.FileHandler("autopatch.log"),   # uncomment to write to .log
                logging.StreamHandler()
            ]
        )
    
    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        pg.setConfigOptions(
            imageAxisOrder="row-major"
        )  # Transposes image in pg.ImageView()
        mainwin = PatchClampUI()
        mainwin.show()
        app.exec_()
        
    start_logger()
    run_app()
