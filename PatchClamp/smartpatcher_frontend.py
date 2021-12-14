# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 15:15:38 2021

@author: TvdrBurgt
"""

import sys
import numpy as np
import logging

from skimage import io

from PyQt5 import QtWidgets
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QDoubleSpinBox, QGroupBox, QLabel, QStackedWidget, QComboBox
import pyqtgraph.exporters
import pyqtgraph as pg

sys.path.append('../')
from PatchClamp.smartpatcher_backend import SmartPatcher
from PatchClamp.ui_patchclamp_sealtest import PatchclampSealTestUI
from PatchClamp.patchclamp import PatchclampSealTest
from HamamatsuCam.camerathread import CameraThread
from PI_ObjectiveMotor.focuser import PIMotor
from PressureDevice.pressurethread import PressureThread
from Micromanipulator.micromanipulator import ScientificaPatchStar
from SampleStageControl.stage import LudlStage
from NIDAQ.constants import MeasurementConstants



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
        self.setWindowTitle("Patchclamp")
        
        """
        -------------------------- Hardware container -------------------------
        """
        hardwareContainer = QGroupBox()
        hardwareLayout = QGridLayout()
        
        # Button to (dis)connect camera
        self.connect_camerathread_button = QPushButton(text="Camera", clicked=self.connect_camerathread)
        self.connect_camerathread_button.setCheckable(True)
        
        # Button to (dis)connect objective motor
        self.connect_objectivemotor_button = QPushButton(text="Objective motor", clicked=self.connect_objectivemotor)
        self.connect_objectivemotor_button.setCheckable(True)
        
        # Button to (dis)connect micromanipulator
        self.connect_micromanipulator_button = QPushButton(text="Micromanipulator", clicked=self.connect_micromanipulator)
        self.connect_micromanipulator_button.setCheckable(True)
        
        # Button to (dis)connect XYstage
        self.connect_XYstage_button = QPushButton(text="XY stage", clicked=self.connect_XYstage)
        self.connect_XYstage_button.setCheckable(True)
        
        # Button to (dis)connect NIDAQ for sealtest
        self.connect_sealtestthread_button = QPushButton(text="NIDAQ/Sealtest", clicked=self.connect_sealtestthread)
        self.connect_sealtestthread_button.setCheckable(True)
        
        # Button to (dis)connect pressure controller
        self.connect_pressurecontroller_button = QPushButton(text="Pressure controller", clicked=self.connect_pressurethread)
        self.connect_pressurecontroller_button.setCheckable(True)
        
        # Button to stop all the running thread
        self.STOP_button = QPushButton(text="STOP", clicked=self.STOP)
        self.STOP_button.setCheckable(True)
        
        hardwareLayout.addWidget(self.connect_camerathread_button, 0, 0, 1, 1)
        hardwareLayout.addWidget(self.connect_objectivemotor_button, 1, 0, 1, 1)
        hardwareLayout.addWidget(self.connect_micromanipulator_button, 2, 0, 1, 1)
        hardwareLayout.addWidget(self.connect_XYstage_button, 3, 0, 1, 1)
        hardwareLayout.addWidget(self.connect_sealtestthread_button, 4, 0, 1, 1)
        hardwareLayout.addWidget(self.connect_pressurecontroller_button, 5, 0, 1, 1)
        hardwareLayout.addWidget(self.STOP_button, 6, 0, 1, 1)
        hardwareContainer.setLayout(hardwareLayout)
        
        """
        ------------------------- Camera view display -------------------------
        """
        liveContainer = QGroupBox()
        liveContainer.setMinimumSize(1200, 600)
        liveLayout = QGridLayout()
        
        # Display to project live camera view
        liveWidget = pg.ImageView()
        liveWidget.ui.roiBtn.hide()
        liveWidget.ui.menuBtn.hide()
        liveWidget.ui.histogram.hide()
        self.liveView = liveWidget.getView()
        self.liveImageItem = liveWidget.getImageItem()
        self.liveImageItem.setAutoDownsample(True)
        
        # Button for pausing camera view
        self.request_pause_button = QPushButton(text="Pause live", clicked=self.toggle_pauselive)
        self.request_pause_button.setCheckable(True)
        request_clear_drawings = QPushButton(text="Clear phantoms", clicked=self.clearROIs)
        
        # Display to project snaphots on
        snapWidget = pg.ImageView()
        snapWidget.ui.roiBtn.hide()
        snapWidget.ui.menuBtn.hide()
        snapWidget.ui.histogram.hide()
        self.snapImageItem = snapWidget.getImageItem()
        self.snapImageItem.setAutoDownsample(True)
        
        # Button for snapshotting
        self.request_snap_button = QPushButton(text="Take snapshot", clicked=self.request_snap)
        
        liveLayout.addWidget(liveWidget, 0, 0, 1, 2)
        liveLayout.addWidget(snapWidget, 0, 2, 1, 2)
        liveLayout.addWidget(self.request_pause_button, 1, 0, 1, 1)
        liveLayout.addWidget(request_clear_drawings, 1, 1, 1, 1)
        liveLayout.addWidget(self.request_snap_button, 1, 2, 1, 2)
        liveContainer.setLayout(liveLayout)
        
        """
        -------------------------- Sensory output display -------------------------
        """
        sensorContainer = QGroupBox()
        sensorContainer.setMinimumSize(400,600)
        sensorLayout = QGridLayout()
        
        sensorWidget = pg.GraphicsLayoutWidget()
        self.algorithm = sensorWidget.addPlot(1, 0, 1, 1)
        self.algorithm.setTitle("Algorithm graph")
        self.algorithm.setLabel("left", units="a.u.")
        self.algorithm.setLabel("bottom", units="a.u.")
        self.algorithmPlot = self.algorithm.plot(pen=(1,3))
        
        currentPlot = sensorWidget.addPlot(2, 0, 1, 1)
        currentPlot.setTitle("Current")
        currentPlot.setLabel("left", units="A")
        currentPlot.setLabel("bottom", text="20 ms")
        self.currentPlot = currentPlot.plot(pen=(2,3))
        
        pressurePlot = sensorWidget.addPlot(3, 0, 1, 1)
        pressurePlot.setTitle("Pressure")
        pressurePlot.setLabel("left", units="mBar")
        pressurePlot.setLabel("bottom", text="time", units="s")
        pressurePlot.setRange(yRange=[-250,250])
        self.pressurePlot = pressurePlot.plot(pen=(3,3))
        
        request_resetplots_button = QPushButton(text="Reset all plots", clicked=self.reset_plots)
        
        sensorLayout.addWidget(sensorWidget, 0, 0, 1, 1)
        sensorLayout.addWidget(request_resetplots_button, 1, 0, 1, 1)
        sensorContainer.setLayout(sensorLayout)
        
        """
        ---------------------- Autopatch control buttons ----------------------
        """
        algorithmContainer = QGroupBox()
        algorithmLayout = QGridLayout()
        
        request_hardcalibrationxy_button = QPushButton(text="Calibrate XY", clicked=self.request_hardcalibration_xy)
        request_hardcalibrationxyz_button = QPushButton(text="Calibrate pixelsize", clicked=self.request_hardcalibration_pixelsize)
        request_prechecks_button = QPushButton(text="Pre-checks", clicked=self.request_prechecks)
        request_selecttarget_button = QPushButton(text="Select target", clicked=self.request_selecttarget)
        request_confirmtarget_button = QPushButton(text="Confirm target", clicked=self.request_confirmtarget)
        request_softcalibration_button = QPushButton(text="Detect tip", clicked=self.request_softcalibration)
        request_target2center_button = QPushButton(text="Move target to center", clicked=self.request_target2center)
        request_pipette2target_button = QPushButton(text="Move pipette to target", clicked=self.request_pipette2target)
        request_autofocustip = QPushButton(text="Autofocus tip", clicked=self.request_autofocustip)
        request_softcalibration_button = QPushButton(text="Detect tip", clicked=self.request_softcalibration)
        
        algorithmLayout.addWidget(request_hardcalibrationxy_button, 0, 0, 1, 1)
        algorithmLayout.addWidget(request_hardcalibrationxyz_button, 1, 0, 1, 1)
        algorithmLayout.addWidget(request_prechecks_button, 0, 1, 2, 1)
        algorithmLayout.addWidget(request_selecttarget_button, 0, 2, 1, 1)
        algorithmLayout.addWidget(request_confirmtarget_button, 1, 2, 1, 1)
        algorithmLayout.addWidget(request_target2center_button, 0, 3, 1, 1)
        algorithmLayout.addWidget(request_pipette2target_button, 1, 3, 1, 1)
        algorithmLayout.addWidget(request_autofocustip, 0, 4, 1, 1)
        algorithmLayout.addWidget(request_softcalibration_button, 1, 4, 1, 1)
        algorithmContainer.setLayout(algorithmLayout)
        
        """
        ----------------------- Sealtest control buttons ----------------------
        """
        sealtestContainer = QGroupBox()
        sealtestLayout = QGridLayout()
        
        self.resistanceLabel = QLabel("Resistance: ")
        self.capacitanceLabel = QLabel("Capacitance: ")
        self.ratioLabel = QLabel("Ratio: ")
        self.membraneVoltLabel = QLabel("Vm: ")
        
        request_gigaseal_button = QPushButton(text="Gigaseal", clicked=self.request_formgigaseal)
        request_breakin_button = QPushButton(text="Break-in", clicked=self.request_breakin)
        # request_gigaseal_button = QPushButton(text="XY grid", clicked=self.request_imagexygrid)
        # request_breakin_button = QPushButton(text="Z stack", clicked=self.request_imagezstack)
        request_zap_button = QPushButton(text="ZAP (don't use!')", clicked=self.mockfunction)
        
        sealtestLayout.addWidget(self.resistanceLabel, 0, 0, 1, 3)
        sealtestLayout.addWidget(self.capacitanceLabel, 0, 3, 1, 3)
        sealtestLayout.addWidget(self.ratioLabel, 0, 6, 1, 3)
        sealtestLayout.addWidget(self.membraneVoltLabel, 0, 9, 1, 3)
        sealtestLayout.addWidget(request_gigaseal_button, 1, 0, 1, 4)
        sealtestLayout.addWidget(request_breakin_button, 1, 4, 1, 4)
        sealtestLayout.addWidget(request_zap_button, 1, 8, 1, 4)
        sealtestContainer.setLayout(sealtestLayout)
        
        """
        ----------------------- Pressure control buttons ----------------------
        """
        pressureContainer = QGroupBox()
        pressureLayout = QGridLayout()
        
        # Live pressure value
        self.pressureLabel = QLabel("Pressure (in mBar):")
        
        # Button to set pressure
        self.set_pressure_button = QDoubleSpinBox(self)
        self.set_pressure_button.setMinimum(-200)
        self.set_pressure_button.setMaximum(200)
        self.set_pressure_button.setDecimals(0)
        self.set_pressure_button.setValue(0)
        self.set_pressure_button.setSingleStep(10)
        
        # Button to release pressure instantaneous
        request_releasepressure_button = QPushButton(text="Release pressure", clicked=self.request_release_pressure)
        
        # Button to record pressure input
        self.request_recordpressure_button = QPushButton(text="Record pressure", clicked=self.request_record_pressure)
        self.request_recordpressure_button.setCheckable(True)
        
        # Button to send pressure to pressure controller
        request_applypressure_button = QPushButton(text="Apply pressure", clicked=self.request_apply_pressure)
        
        pressureLayout.addWidget(self.pressureLabel, 0, 0, 1, 2)
        pressureLayout.addWidget(self.set_pressure_button, 0, 2, 1, 1)
        pressureLayout.addWidget(request_releasepressure_button, 1, 0, 1, 1)
        pressureLayout.addWidget(self.request_recordpressure_button, 1, 1, 1, 1)
        pressureLayout.addWidget(request_applypressure_button, 1, 2, 1, 1)
        pressureContainer.setLayout(pressureLayout)
        
        """
        ---------------------- Add widgets and set Layout ---------------------
        """
        master = QGridLayout()
        master.addWidget(hardwareContainer, 0, 0, 1, 1)
        master.addWidget(liveContainer, 0, 1, 1, 2)
        master.addWidget(sensorContainer, 0, 3, 1, 1)
        master.addWidget(algorithmContainer, 1, 0, 1, 2)
        master.addWidget(sealtestContainer, 1, 2, 1, 1)
        master.addWidget(pressureContainer, 1, 3, 1, 1)
        
        autopatcher = QGroupBox()
        autopatcher.setLayout(master)
        
        """
        -------------------- Stack automatic/manual GUI's  --------------------
        """
        self.stackedWidget = QStackedWidget()
        self.stackedWidget.addWidget(autopatcher)
        self.stackedWidget.addWidget(PatchclampSealTestUI())
        
        pageComboBox = QComboBox()
        pageComboBox.addItem(str("Automatic patcher"))
        pageComboBox.addItem(str("Manual patcher"))
        pageComboBox.activated.connect(self.stackedWidget.setCurrentIndex)
        pageComboBox.activated.connect(self.resize_QStack)
        pageComboBox.AdjustToContents = True
        
        layout = QGridLayout()
        layout.addWidget(pageComboBox, 0,0,1,1)
        layout.addWidget(self.stackedWidget, 1,0,1,1)
        self.setLayout(layout)
        
        
        """
        =======================================================================
        ------------------------ Start up roi manager -------------------------
        =======================================================================
        """
        
        self.roimanager = ROIManagerGUI(offset=len(self.liveView.addedItems))
        
        """
        =======================================================================
        -------------- Start up backend and connect signals/slots--------------
        =======================================================================
        """
        
        self.backend = SmartPatcher()
        self.backend.worker.draw.connect(self.draw_roi)
        self.backend.worker.graph.connect(self.update_algorithmplot)
        self.backend.worker.progress.connect(self.mocksignal)
        self.backend.worker.finished.connect(self.mocksignal)
        
        """
        =======================================================================
        ----------------------------- End of GUI ------------------------------
        =======================================================================
        """
        
        
    def mocksignal(self):
        print('Algorithm finished')
        logging.info('QThread isFinished: ' + str(self.backend.thread.isFinished()))
        logging.info('QThread isRunning: ' + str(self.backend.thread.isRunning()))
        
        
    def mockfunction(self):
        print("Button pushed")
        self.backend.request(name='mockworker')
        
    
    def resize_QStack(self):
        activelayer = self.stackedWidget.currentIndex()
        if activelayer == 1:
            self.setFixedSize(400,750)
        else:
            self.setFixedSize(1800,800)
            
    
    def connect_micromanipulator(self):
        """
        We initiate the micromanipulator by creating the ScientificaPatchStar
        serial object. Creating the object object automatically opens the USB
        port that the device is connected to, similarly, deleting the object
        stops micromanipulator movement and closes the port.
        """
        if self.connect_micromanipulator_button.isChecked():
            micromanipulator = ScientificaPatchStar(address='COM16', baud=38400)
            
            self.backend.micromanipulator = micromanipulator
        else:
            del self.backend.micromanipulator
    
    
    def connect_XYstage(self):
        """
        We initiate the XYstage (sample stage) by creating the LudlStage object
        Creating the object does not automatically open the USB port to which
        the device is connected to. Deleting the object stops stage movement.
        """
        if self.connect_XYstage_button.isChecked():
            ludlStage = LudlStage('COM12')
            
            self.backend.XYstage = ludlStage
        else:
            del self.backend.XYstage
    
    
    def connect_objectivemotor(self):
        """
        We initiate the objective motor by creating the PIMotor object. This
        takes time in which the GUI freezes, this issue can be solved by
        passing the objective motor instance from Tupolev/Fiumicino as an
        objective_motor_handle. Deleting the object disconnects the objective
        motor.
        """
        if self.connect_objectivemotor_button.isChecked():
            objectivemotor = PIMotor(objective_motor_handle=None)
            self.backend.objectivemotor = objectivemotor
        else:
            del self.backend.objectivemotor
    
    
    def connect_camerathread(self):
        """
        We initiate the camera by creating the CameraThread object, then we
        connect signals with slots to govern inter-thread communication, i.e.
        to recieve images to display. Afterwards we move the camera thread to
        the backend where the thread is immediately started.
        """
        if self.connect_camerathread_button.isChecked():
            camerathread = CameraThread(camerahandle=None)
            
            self.signal_camera_live = camerathread.livesignal
            self.signal_camera_snap = camerathread.snapsignal
            self.signal_camera_live.connect(self.update_live)
            self.signal_camera_snap.connect(self.update_snap)
            
            self.backend.camerathread = camerathread
        else:
            del self.backend.camerathread
            del self.signal_camera_live
            del self.signal_camera_snap
    
    
    def connect_sealtestthread(self):
        """
        We initiate the sealtest thread by creating the SealTestThread object,
        then we connect the measurement signal to update the graph for voltage
        and current. Afterwards we move the sealtest thread to the backend
        where the thread is immediately started.
        """
        logging.info('connect sealtestthread button pushed')
        if self.connect_sealtestthread_button.isChecked():
            sealtestthread = PatchclampSealTest()
            
            self.signal_sealtest = sealtestthread.measurementThread.measurement
            self.signal_sealtest.connect(self.update_currentvoltageplot)
            
            self.backend.sealtestthread = sealtestthread
        else:
            del self.backend.sealtestthread
            del self.signal_sealtest
    
    
    def connect_pressurethread(self):
        """
        We initiate the pressure thread by creating the PressureThread object,
        then we connect the measurement signal to update the pressure evolution
        graph. Afterwards we move the sealtest thread to the backend
        where the thread is immediately started.
        """
        logging.info('connect pressurethread button pushed')
        if self.connect_pressurecontroller_button.isChecked():
            pressurethread = PressureThread(pressurecontroller_handle=None)
            
            self.signal_pressure = pressurethread.measurement
            self.signal_pressure.connect(self.update_pressureplot)
            
            self.backend.pressurethread = pressurethread
        else:
            del self.backend.pressurethread
            del self.signal_pressure
    
    
    def toggle_pauselive(self):
        if hasattr(self, 'signal_camera_live'):
            if self.request_pause_button.isChecked():
                try:
                    self.signal_camera_live.disconnect()
                except TypeError:
                    pass
                self.request_pause_button.setChecked(True)
            else:
                try:
                    self.signal_camera_live.disconnect()
                except TypeError:
                    pass
                self.signal_camera_live.connect(self.update_live)
                self.request_pause_button.setChecked(False)
        else:
            self.request_pause_button.setChecked(False)
        
        
    def request_snap(self):
        I = self.backend.camerathread.snap()
        io.imsave(self.backend.save_directory+'.tif', I, check_contrast=False)
        self.update_snap(I)
        # if self.backend.camerathread != None:
        #     I = self.backend.camerathread.snap()
        #     io.imsave(self.backend.save_directory+'.tif', I, check_contrast=False)
        #     self.update_snap(I)
        # else:
        #     raise ValueError('no camera connected')
    
    def request_release_pressure(self):
        self.backend.pressurethread.set_pressure_stop_waveform(0)
    
    def request_record_pressure(self):
        if self.request_recordpressure_button.isChecked():
            self.backend.pressurethread.isrecording = True
        else:
            self.backend.pressurethread.isrecording = False
    
    def request_apply_pressure(self):
        target_pressure = self.set_pressure_button.value()
        self.backend.pressurethread.set_pressure_stop_waveform(target_pressure)
    
    def request_hardcalibration_xy(self):
        self.backend.request(name='hardcalibration', mode='XY')
        
    def request_hardcalibration_pixelsize(self):
        self.backend.request(name='hardcalibration', mode='pixelsize')
    
    def request_prechecks(self):
        self.backend.request(name='prechecks')
    
    def request_softcalibration(self):
        self.backend.request(name='softcalibration')
    
    def request_target2center(self):
        self.backend.request(name='target2center')
    
    def request_pipette2target(self):
        self.backend.request(name='pipette2target')
    
    def request_autofocustip(self):
        # Clear the plot and update all labels
        self.algorithmPlot.setData()
        self.algorithm.setTitle('Autofocus')
        self.algorithm.setLabel("left", text="sharpness", units="a.u.")
        self.algorithm.setLabel("bottom", text="height", units="um")
        
        self.backend.request(name='autofocustip')
    
    def request_selecttarget(self):
        """
        The user drags a circular ROI on top of the target cell. The ROI center
        is the target. We first check if a target already exists, if so, we
        recycle it. If the target ROI got removed then we place it back to its
        last known position.
        """
        self.request_pause_button.setChecked(True)
        self.toggle_pauselive()
        
        coords = self.backend.target_coordinates
        if all(values is None for values in coords):
            coords = (0,0)
        else:
            coords = (coords[0]-60,coords[1]-60)
            
        if not self.roimanager.contains('target'):
            target = pg.CircleROI(pos=coords, radius=60, movable=True, pen=QPen(QColor(255,255,0), 0))
            target.setZValue(10)
            self.roimanager.addROI('target')
            self.liveView.addItem(target)
        else:
            idx = self.roimanager.giveROIindex('target')[-1]
            self.liveView.addedItems[idx].translatable = True
            self.liveView.addedItems[idx].setPen(QPen(QColor(255,255,0), 0))
        
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
            self.backend.target_coordinates = np.array([x,y,None])
    
    def request_formgigaseal(self):
        # Clear the algorithm plot and update all labels
        self.algorithmPlot.setData()
        self.algorithm.setTitle('Pipette resistance')
        self.algorithm.setLabel("left", text="R", units='Ohm')
        self.algorithm.setLabel("bottom", text="time", units="")
        
        self.backend.request(name='gigaseal')
    
    def request_breakin(self):
        self.backend.request(name='breakin')
    
    def request_imagexygrid(self):
        self.backend.request(name='request_imagexygrid')
    
    def request_imagezstack(self):
        self.backend.request(name='request_imagezstack')
    
    
    def draw_roi(self, *args):
        label = args[0][0]
        if label == 'cross':
            xpos,ypos = args[0][1:3]
            vertical = pg.ROI(pos=(xpos,ypos-15), size=[1,30], pen=(1,3))
            horizontal = pg.ROI(pos=(xpos-15,ypos), size=[30,1], pen=(1,3))
            vertical.setZValue(10)
            horizontal.setZValue(10)
            self.roimanager.addROI('cross' + '_vertical')
            self.roimanager.addROI('cross' + '_horizontal')
            self.liveView.addItem(vertical)
            self.liveView.addItem(horizontal)
        elif label == 'calibrationline':
            xpos,ypos,orientation = args[0][1:4]
            if len(self.roimanager.giveROIindex('calibrationline')) == 0:
                P = (0,3)
            elif len(self.roimanager.giveROIindex('calibrationline')) == 1:
                P = (2,3)
            else:
                P = (1,3)
            horizontal = pg.ROI(pos=(xpos-15,ypos), angle=orientation, size=[500,1], pen=P)
            horizontal.setZValue(10)
            self.roimanager.addROI('calibrationline')
            self.liveView.addItem(horizontal)
        elif label == 'image':
            img = args[0][1]
            self.update_snap(img)
        elif label == 'target':
            dx,dy = args[0][1:3]
            idx = self.roimanager.giveROIindex('target')[-1]
            x,y = self.liveView.addedItems[idx].state['pos']
            self.liveView.addedItems[idx].setPos([x-dx,y-dy])
        elif label == 'algorithm threshold':
            upper_threshold = args[0][1]
            lower_threshold = args[0][2]
            self.algorithm.addItem(pg.InfiniteLine(pos=upper_threshold, angle=0))
            self.algorithm.addItem(pg.InfiniteLine(pos=lower_threshold, angle=0))
        elif label == 'remove algorithm threshold':
            for item in self.algorithm.items[1::]:
                self.algorithm.removeItem(item)
        else:
            print(label + ' is not a known draw-label')
    
    
    def clearROIs(self):
        self.roimanager.removeallROIs()
        for roi in self.liveView.addedItems[self.roimanager.offset::]:
            self.liveView.removeItem(roi)
    
    
    def update_live(self, image):
        self.liveImageItem.setImage(image)
        
        
    def update_snap(self, image):
        self.snapImageItem.setImage(image)
    
    
    def update_algorithmplot(self, data):
        if data.ndim == 1:
            xs = data
            self.algorithmPlot.setData(xs)
        else:
            xs = data[0,:]
            ys = data[1,:]
            self.algorithmPlot.setData(xs,ys)
    
    def update_pressureplot(self, data):
        self.backend._pressure_append_(data[0], data[1])
        pressuredata = self.backend.pressure
        
        self.pressurePlot.setData(pressuredata[1], pressuredata[0])
        self.pressureLabel.setText("Pressure (in mBar): %.1f" % pressuredata[0,-1])
    
    def update_currentvoltageplot(self, voltOut, curOut):
        self.backend._voltage_append_(voltOut / 10)
        self.backend._current_append_(curOut / 1 / (100*10**6))
        # voltage = self.backend.voltage
        current = self.backend.current
        
        self.currentPlot.setData(current)
        
        """ to calculate capacitance and resistance """
        self.updateLabels(self.backend.current, self.backend.voltage)
    
    
    def updateLabels(self, curOut, voltOut):
        """Update the resistance and capacitance labels.
        http://scipy-lectures.org/intro/scipy/auto_examples/plot_curve_fit.html
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        This function was taken from lhuismans and modified by TvdrBurgt.
        """
        constants = MeasurementConstants()
        sampPerCyc = int(constants.patchSealSampRate / constants.patchSealFreq)

        try:
            curOutCyc = curOut.reshape(int(curOut.size / sampPerCyc), sampPerCyc)
            curData = np.mean(curOutCyc, axis=0)
        except:
            curData = curOut

        voltData = voltOut
        try:
            # Computing resistance
            tres = np.mean(voltData)
            dV = np.mean(voltData[voltData > tres]) - np.mean(voltData[voltData < tres])  # Computing the voltage difference
            dIss = np.mean(curData[int(np.floor(0.15 * sampPerCyc)) : int(np.floor(sampPerCyc / 2)) - 2]) - np.mean(curData[int(np.floor(0.65 * sampPerCyc)) : sampPerCyc - 2])  # Computing the current distance
            membraneResistance = dV / (dIss * 1000000)  # Ohms law (MegaOhm)
            self.resistanceLabel.setText("Resistance:  %.4f M\u03A9" % membraneResistance)
            R_to_append = membraneResistance*1e6

            estimated_size_resistance = 10000 / (membraneResistance * 1000000)  # The resistance of a typical patch of membrane, RM is 10000 Omega/{cm}^2
        except:
            self.resistanceLabel.setText("Resistance:  %s" % "NaN")
            R_to_append = np.nan

        try:
            measured_vlotage = np.mean(voltData) * 1000
            self.membraneVoltLabel.setText("Vm:  %.2f mV" % measured_vlotage)
            self.membraneVoltLabel.setStyleSheet("color: red")
        except:
            self.membraneVoltLabel.setText("Vm:  %s" % "NaN")
        try:
            # Computing capacitance
            points = 10
            maxCur = np.amax(curData)
            maxCurIndex = np.where(curData == maxCur)[0][0]
            curFit = curData[int(maxCurIndex+1) : int(maxCurIndex+1+points-1)] - \
                0.5*(np.mean(curData[int(np.floor(0.15*sampPerCyc)) : int(np.floor(sampPerCyc/2)) - 2]) + \
                     np.mean(curData[int(np.floor(0.65*sampPerCyc)) : sampPerCyc-2]))
            timepoints = 1000*np.arange(3, points - 1 + 3) / constants.patchSealSampRate
            # Fitting the data to an exponential of the form y=a*exp(-b*x) where b = 1/tau and tau = RC
            # I(t)=I0*e^−t/τ, y=a*exp(-b*x), get log of both sides:log y = -bx + log a
            fit = np.polyfit(timepoints, curFit, 1)  # Converting the exponential to a linear function and fitting it
            # Extracting data
            current = fit[0]
            resistance = dV * 1000 / current / 2  # Getting the resistance
            tau = -1 / fit[1]
            capacitance = 1000 * tau / resistance
            self.capacitanceLabel.setText("Capacitance:  %.4f" % capacitance)
            C_to_append = capacitance

            estimated_size_capacitance = capacitance * (10 ** -12) * (10 ** 6)

            if estimated_size_capacitance > estimated_size_resistance:
                estimated_ratio = (estimated_size_capacitance / estimated_size_resistance)
            else:
                estimated_ratio = (estimated_size_resistance / estimated_size_capacitance)

            self.ratioLabel.setText("Ratio:  %.4f" % estimated_ratio)  # http://www.cnbc.cmu.edu/~bard/passive2/node5.html

        except:
            self.capacitanceLabel.setText("Capacitance:  %s" % "NaN")
            C_to_append = np.nan
            self.ratioLabel.setText("Ratio:  %s" % "NaN")
        
        # store resistance and capacitance values in backend
        self.backend._resistance_capacitance_append_(R_to_append, C_to_append)
    
    
    def reset_plots(self):
        del self.backend.current
        del self.backend.voltage
        del self.backend.pressure
        self.algorithmPlot.setData([0])
        self.pressurePlot.setData([0])
        self.currentPlot.setData([0])
    
    
    def STOP(self):
        if self.STOP_button.isChecked():
            self.backend.STOP = True
        else:
            self.backend.STOP = False
        
    
    def moveEvent(self, event):
        super(PatchClampUI, self).moveEvent(event)
        self.request_pause_button.setChecked(True)
        self.toggle_pauselive()
        event.accept()
    
    
    def closeEvent(self, event):
        """ Close event
        This method is called when the GUI is shut down. First we need to stop
        the threads that are still running, then we disconnect all hardware to
        be reused in the main widget, only then we accept the close event.
        and quit the widget.
        """
        self.backend.STOP = True
        
        try:
            del self.backend.camerathread
        except AttributeError:
            pass
        try:
            del self.backend.micromanipulator
        except AttributeError:
            pass
        try:
            del self.backend.sealtestthread
        except AttributeError:
            pass
        try:
            del self.backend.pressurethread
        except AttributeError:
            pass
        
        event.accept()
        
        # Frees the console by quitting the application entirely. (uncomment if running at home)
        # Will not let you re-run the GUI somehow. (comment if running in the lab, or with Fiumicino)
        # QtWidgets.QApplication.quit()
        
        


class ROIManagerGUI:
    def __init__(self, offset):
        self.offset = offset
        self.ROInumber = 0
        self.ROIdictionary = {}
    
    def giveROIindex(self, name):
        if self.contains(name):
            index_array = [x+self.offset for x in self.ROIdictionary[name][:]]
        else:
            index_array = []
        return index_array
    
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
                
    def removeallROIs(self):
        name = list(self.ROIdictionary)
        for name in name:
            self.removeROI(name, args='all')
                
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
    