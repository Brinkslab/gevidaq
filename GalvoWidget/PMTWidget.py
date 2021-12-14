# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 13:34:56 2020

@author: xinmeng
"""
from __future__ import division
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPoint, QRect, QObject
from PyQt5.QtGui import QColor, QPen, QPixmap, QIcon, QTextCursor, QFont

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
)

import pyqtgraph as pg
import sys
import numpy as np
from PIL import Image
from datetime import datetime
from pyqtgraph import PlotDataItem
import os
import matplotlib.pyplot as plt

# Ensure that the Widget can be run either independently or as part of Tupolev.
if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname + "/../")

from GeneralUsage.ThreadingFunc import run_in_thread
from NIDAQ.constants import HardwareConstants
from GalvoWidget.pmt_thread import pmtimagingTest, pmtimagingTest_contour
from GalvoWidget.GalvoScan_backend import PMT_zscan
from NIDAQ.DAQoperator import DAQmission
import StylishQT


class PMTWidgetUI(QWidget):

    #    waveforms_generated = pyqtSignal(object, object, list, int)
    SignalForContourScanning = pyqtSignal(int, int, int, np.ndarray, np.ndarray)
    MessageBack = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #        os.chdir('./')# Set directory to current folder.
        self.setFont(QFont("Arial"))

        self.setMinimumSize(1200, 850)
        self.setWindowTitle("PMTWidget")
        self.layout = QGridLayout(self)
        # ------------------------Initiating class-------------------
        self.pmtTest = pmtimagingTest()
        self.pmtTest_contour = pmtimagingTest_contour()

        self.savedirectory = r"M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Octoscope\pmt_image_default_dump"
        self.prefixtextboxtext = "_fromGalvoWidget"
        
        self.contour_ROI_signals_dict = {}
        self.contour_ROI_handles_dict = {}
        
        self.clicked_points_list = []
        self.flag_is_drawing = False
        # **************************************************************************************************************************************
        # --------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------GUI for PMT tab------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------------------------------
        # **************************************************************************************************************************************
        pmtimageContainer = StylishQT.roundQGroupBox(title="PMT image")
        self.pmtimageLayout = QGridLayout()
        
        self.pmtvideoWidget = pg.ImageView()
        self.pmtvideoWidget.ui.roiBtn.hide()
        self.pmtvideoWidget.ui.menuBtn.hide()
        self.pmtvideoWidget.resize(400, 400)
        self.pmtimageLayout.addWidget(self.pmtvideoWidget, 0, 0)

        pmtroiContainer = StylishQT.roundQGroupBox(title="PMT ROI")
        self.pmtimageroiLayout = QGridLayout()

        self.pmt_roiwidget = pg.GraphicsLayoutWidget()
        self.pmt_roiwidget.resize(150, 150)
        self.pmt_roiwidget.addLabel("ROI", row=0, col=0)
        
        
        # --------------------------- create ROI ------------------------------
        self.vb_2 = self.pmt_roiwidget.addViewBox(
            row=1, col=0, lockAspect=True, colspan=1
        )
        self.vb_2.name = "ROI"

        self.pmtimgroi = pg.ImageItem()
        self.vb_2.addItem(self.pmtimgroi)
        # self.roi = pg.RectROI([20, 20], [20, 20], pen=(0,9))
        # r1 = QRectF(0, 0, 895, 500)
        self.ROIpen = QPen()  # creates a default pen
        self.ROIpen.setStyle(Qt.DashDotLine)
        self.ROIpen.setWidth(1)
        self.ROIpen.setBrush(QColor(0, 161, 255))
        
        self.roi = pg.PolyLineROI(
            [[0, 0], [80, 0], [80, 80], [0, 80]], closed=True, pen=self.ROIpen
        )  # , maxBounds=r1
        # self.roi.addRotateHandle([40,0], [0.5, 0.5])
        self.roi.sigHoverEvent.connect(
            lambda: self.show_handle_num(self.roi)
        )  # update handle numbers
        
        self.pmtvb = self.pmtvideoWidget.getView()
        
        self.pmtimageitem = self.pmtvideoWidget.getImageItem()
        # self.pmtvb.addItem(self.roi)  # add ROIs to main image
            
        self.pmtvb.scene().sigMouseClicked.connect(self.generate_poly_roi)

        pmtimageContainer.setMinimumWidth(850)
        pmtroiContainer.setFixedHeight(320)
        #        pmtroiContainer.setMaximumWidth(300)

        pmtimageContainer.setLayout(self.pmtimageLayout)
        pmtroiContainer.setLayout(self.pmtimageroiLayout)

        # ----------------------------Contour-----------------------------------
        pmtContourContainer = StylishQT.roundQGroupBox(title="Contour selection")
        # pmtContourContainer.setFixedWidth(280)
        self.pmtContourLayout = QGridLayout()
        # contour_Description = QLabel("Handle number updates when parking mouse cursor upon ROI. Points in contour are divided evenly between handles.")
        # contour_Description.setStyleSheet('color: blue')
        # self.pmtContourLayout.addWidget(contour_Description,0,0)

        self.pmt_handlenum_Label = QLabel("Handle number: ")
        self.pmtContourLayout.addWidget(self.pmt_handlenum_Label, 1, 0)

        self.contour_strategy = QComboBox()
        self.contour_strategy.addItems(["Evenly between", "Uniform"])
        self.contour_strategy.setToolTip("Even in-between: points evenly distribute inbetween handles; Uniform: evenly distribute regardless of handles")
        self.pmtContourLayout.addWidget(self.contour_strategy, 1, 1)

        self.pointsinContour = QSpinBox(self)
        self.pointsinContour.setMinimum(1)
        self.pointsinContour.setMaximum(1000)
        self.pointsinContour.setValue(100)
        self.pointsinContour.setSingleStep(100)
        self.pmtContourLayout.addWidget(self.pointsinContour, 1, 3)
        self.pmtContourLayout.addWidget(QLabel("Points in contour:"), 1, 2)

        self.contour_samprate = QSpinBox(self)
        self.contour_samprate.setMinimum(0)
        self.contour_samprate.setMaximum(1000000)
        self.contour_samprate.setValue(50000)
        self.contour_samprate.setSingleStep(50000)
        self.pmtContourLayout.addWidget(self.contour_samprate, 2, 1)
        self.pmtContourLayout.addWidget(QLabel("Sampling rate:"), 2, 0)
        
        self.pmtContourLayout.addWidget(QLabel("Contour index:"), 3, 0)
        self.roi_index_spinbox = QSpinBox(self)
        self.roi_index_spinbox.setMinimum(1)
        self.roi_index_spinbox.setMaximum(20)
        self.roi_index_spinbox.setValue(1)
        self.roi_index_spinbox.setSingleStep(1)
        self.pmtContourLayout.addWidget(self.roi_index_spinbox, 3, 1)
        
        self.go_to_first_handle_button = StylishQT.GeneralFancyButton(label = "Go 1st point")
        self.go_to_first_handle_button.setFixedHeight(32)
        # self.pmtContourLayout.addWidget(self.go_to_first_handle_button, 4, 1)
        self.go_to_first_handle_button.clicked.connect(self.go_to_first_point)
        self.go_to_first_handle_button.setToolTip("Set gavlo initial positions in advance")
        
        ROI_interaction_tips = QLabel("Hover for tips. Key F:en/disable drawing ROI")
        ROI_interaction_tips.setToolTip("Left drag moves the ROI\n\
Left drag + Ctrl moves the ROI with position snapping\n\
Left drag + Alt rotates the ROI\n\
Left drag + Alt + Ctrl rotates the ROI with angle snapping\n\
Left drag + Shift scales the ROI\n\
Left drag + Shift + Ctrl scales the ROI with size snapping")
        self.pmtContourLayout.addWidget(ROI_interaction_tips, 4, 0, 1, 2)
        
        self.regenerate_roi_handle_button = StylishQT.GeneralFancyButton(label = "Regain ROI")
        self.regenerate_roi_handle_button.setFixedHeight(32)
        self.pmtContourLayout.addWidget(self.regenerate_roi_handle_button, 3, 2)
        self.regenerate_roi_handle_button.clicked.connect(self.regenerate_roi_handles)
        
        self.reset_roi_handle_button = StylishQT.GeneralFancyButton(label = "Reset handles")
        self.reset_roi_handle_button.setFixedHeight(32)
        self.pmtContourLayout.addWidget(self.reset_roi_handle_button, 3, 3)
        self.reset_roi_handle_button.clicked.connect(self.reset_roi_handles)
        
        # Button to add roi to stack
        self.add_roi_to_stack_button = StylishQT.addButton()
        self.add_roi_to_stack_button.setFixedHeight(32)
        self.pmtContourLayout.addWidget(self.add_roi_to_stack_button, 4, 2)
        self.add_roi_to_stack_button.clicked.connect(self.add_coordinates_to_list)
        
        self.del_roi_in_stack_button = StylishQT.stop_deleteButton()
        self.del_roi_in_stack_button.setFixedHeight(32)
        self.del_roi_in_stack_button.clicked.connect(self.del_coordinates_from_list)
        self.pmtContourLayout.addWidget(self.del_roi_in_stack_button, 4, 3)
    
        self.reset_roi_stack_button = StylishQT.cleanButton("Clear")
        self.reset_roi_stack_button.setFixedHeight(32)
        self.reset_roi_stack_button.setToolTip("Clear ROI info")
        self.pmtContourLayout.addWidget(self.reset_roi_stack_button, 5, 0)
        self.reset_roi_stack_button.clicked.connect(self.reset_coordinates_dict)
        
        self.generate_contour_sacn = StylishQT.generateButton()
        self.pmtContourLayout.addWidget(self.generate_contour_sacn, 5, 1)
        self.generate_contour_sacn.clicked.connect(lambda: self.generate_final_contour_signals())

        self.do_contour_sacn = StylishQT.runButton("Contour")
        self.do_contour_sacn.setFixedHeight(32)
        self.pmtContourLayout.addWidget(self.do_contour_sacn, 5, 2)
        self.do_contour_sacn.clicked.connect(
            lambda: self.buttonenabled("contourscan", "start")
        )
        self.do_contour_sacn.clicked.connect(lambda: self.measure_pmt_contourscan())

        self.stopButton_contour = StylishQT.stop_deleteButton()
        self.stopButton_contour.setFixedHeight(32)
        self.stopButton_contour.clicked.connect(
            lambda: self.buttonenabled("contourscan", "stop")
        )
        self.stopButton_contour.clicked.connect(
            lambda: self.stopMeasurement_pmt_contour()
        )
        self.stopButton_contour.setEnabled(False)
        self.pmtContourLayout.addWidget(self.stopButton_contour, 5, 3)

        pmtContourContainer.setLayout(self.pmtContourLayout)

        # ----------------------------Control-----------------------------------
        # controlContainer = StylishQT.roundQGroupBox(title = "Galvo Scanning Panel")
        # controlContainer.setFixedWidth(280)
        self.scanning_tabs = QTabWidget()
        self.scanning_tabs.setFixedWidth(280)
        self.scanning_tabs.setFixedHeight(320)
        
        # ---------------------------- Continuous scanning -----------------------------------
        Continuous_widget = QWidget()
        controlLayout = QGridLayout()

        self.pmt_fps_Label = QLabel("Per frame: ")
        controlLayout.addWidget(self.pmt_fps_Label, 5, 0)

        self.saveButton_pmt = StylishQT.saveButton()
        self.saveButton_pmt.clicked.connect(lambda: self.saveimage_pmt())
        controlLayout.addWidget(self.saveButton_pmt, 5, 1)

        self.startButton_pmt = StylishQT.runButton("")
        self.startButton_pmt.setFixedHeight(32)
        self.startButton_pmt.setCheckable(True)
        self.startButton_pmt.clicked.connect(
            lambda: self.buttonenabled("rasterscan", "start")
        )
        self.startButton_pmt.clicked.connect(lambda: self.measure_pmt())

        controlLayout.addWidget(self.startButton_pmt, 6, 0)

        self.stopButton = StylishQT.stop_deleteButton()
        self.stopButton.setFixedHeight(32)
        self.stopButton.clicked.connect(
            lambda: self.buttonenabled("rasterscan", "stop")
        )
        self.stopButton.clicked.connect(lambda: self.stopMeasurement_pmt())
        self.stopButton.setEnabled(False)
        controlLayout.addWidget(self.stopButton, 6, 1)

        # ---------------------Galvo scanning-----------------------------------
        self.continuous_scanning_sr_spinbox = QSpinBox(self)
        self.continuous_scanning_sr_spinbox.setMinimum(0)
        self.continuous_scanning_sr_spinbox.setMaximum(1000000)
        self.continuous_scanning_sr_spinbox.setValue(250000)
        self.continuous_scanning_sr_spinbox.setSingleStep(100000)
        controlLayout.addWidget(self.continuous_scanning_sr_spinbox, 1, 1)
        controlLayout.addWidget(QLabel("Sampling rate:"), 1, 0)

        # controlLayout.addWidget(QLabel("Galvo raster scanning : "), 1, 0)
        self.continuous_scanning_Vrange_spinbox = QSpinBox(self)
        self.continuous_scanning_Vrange_spinbox.setMinimum(-10)
        self.continuous_scanning_Vrange_spinbox.setMaximum(10)
        self.continuous_scanning_Vrange_spinbox.setValue(3)
        self.continuous_scanning_Vrange_spinbox.setSingleStep(1)
        controlLayout.addWidget(self.continuous_scanning_Vrange_spinbox, 2, 1)
        controlLayout.addWidget(QLabel("Volt range:"), 2, 0)

        self.Scanning_pixel_num_combobox = QSpinBox(self)
        self.Scanning_pixel_num_combobox.setMinimum(0)
        self.Scanning_pixel_num_combobox.setMaximum(1000)
        self.Scanning_pixel_num_combobox.setValue(500)
        self.Scanning_pixel_num_combobox.setSingleStep(244)
        controlLayout.addWidget(self.Scanning_pixel_num_combobox, 3, 1)
        controlLayout.addWidget(QLabel("Pixel number:"), 3, 0)

        self.continuous_scanning_average_spinbox = QSpinBox(self)
        self.continuous_scanning_average_spinbox.setMinimum(1)
        self.continuous_scanning_average_spinbox.setMaximum(20)
        self.continuous_scanning_average_spinbox.setValue(1)
        self.continuous_scanning_average_spinbox.setSingleStep(1)
        controlLayout.addWidget(self.continuous_scanning_average_spinbox, 4, 1)
        controlLayout.addWidget(QLabel("average over:"), 4, 0)

        Continuous_widget.setLayout(controlLayout)

        # -------------------------- stack scanning ----------------------------
        Zstack_widget = QWidget()
        Zstack_Layout = QGridLayout()

        self.stack_scanning_sampling_rate_spinbox = QSpinBox(self)
        self.stack_scanning_sampling_rate_spinbox.setMinimum(0)
        self.stack_scanning_sampling_rate_spinbox.setMaximum(1000000)
        self.stack_scanning_sampling_rate_spinbox.setValue(250000)
        self.stack_scanning_sampling_rate_spinbox.setSingleStep(100000)
        Zstack_Layout.addWidget(self.stack_scanning_sampling_rate_spinbox, 1, 1)
        Zstack_Layout.addWidget(QLabel("Sampling rate:"), 1, 0)

        self.stack_scanning_Vrange_spinbox = QSpinBox(self)
        self.stack_scanning_Vrange_spinbox.setMinimum(-10)
        self.stack_scanning_Vrange_spinbox.setMaximum(10)
        self.stack_scanning_Vrange_spinbox.setValue(3)
        self.stack_scanning_Vrange_spinbox.setSingleStep(1)
        Zstack_Layout.addWidget(self.stack_scanning_Vrange_spinbox, 2, 1)
        Zstack_Layout.addWidget(QLabel("Volt range:"), 2, 0)

        self.stack_scanning_Pnumber_spinbox = QSpinBox(self)
        self.stack_scanning_Pnumber_spinbox.setMinimum(0)
        self.stack_scanning_Pnumber_spinbox.setMaximum(1000)
        self.stack_scanning_Pnumber_spinbox.setValue(500)
        self.stack_scanning_Pnumber_spinbox.setSingleStep(244)
        Zstack_Layout.addWidget(self.stack_scanning_Pnumber_spinbox, 3, 1)
        Zstack_Layout.addWidget(QLabel("Pixel number:"), 3, 0)

        self.stack_scanning_Avgnumber_spinbox = QSpinBox(self)
        self.stack_scanning_Avgnumber_spinbox.setMinimum(1)
        self.stack_scanning_Avgnumber_spinbox.setMaximum(20)
        self.stack_scanning_Avgnumber_spinbox.setValue(1)
        self.stack_scanning_Avgnumber_spinbox.setSingleStep(1)
        Zstack_Layout.addWidget(self.stack_scanning_Avgnumber_spinbox, 4, 1)
        Zstack_Layout.addWidget(QLabel("average over:"), 4, 0)

        self.stack_scanning_stepsize_spinbox = QDoubleSpinBox(self)
        self.stack_scanning_stepsize_spinbox.setMinimum(-10000)
        self.stack_scanning_stepsize_spinbox.setMaximum(10000)
        self.stack_scanning_stepsize_spinbox.setDecimals(6)
        self.stack_scanning_stepsize_spinbox.setSingleStep(0.001)
        self.stack_scanning_stepsize_spinbox.setValue(0.004)
        Zstack_Layout.addWidget(self.stack_scanning_stepsize_spinbox, 5, 1)
        Zstack_Layout.addWidget(QLabel("Step size(mm):"), 5, 0)

        self.stack_scanning_depth_spinbox = QDoubleSpinBox(self)
        self.stack_scanning_depth_spinbox.setMinimum(-10000)
        self.stack_scanning_depth_spinbox.setMaximum(10000)
        self.stack_scanning_depth_spinbox.setDecimals(6)
        self.stack_scanning_depth_spinbox.setSingleStep(0.001)
        self.stack_scanning_depth_spinbox.setValue(0.012)
        Zstack_Layout.addWidget(self.stack_scanning_depth_spinbox, 6, 1)
        
        depth_label = QLabel("Depth(mm):")
        Zstack_Layout.addWidget(depth_label, 6, 0)
        depth_label.setToolTip(
            "In case of not changing z-position, set here to 0."
        )

        self.startButton_stack_scanning = StylishQT.runButton("")
        self.startButton_stack_scanning.setFixedHeight(32)
        self.startButton_stack_scanning.setCheckable(True)
        self.startButton_stack_scanning.clicked.connect(
            lambda: self.buttonenabled("stackscan", "start")
        )
        self.startButton_stack_scanning.clicked.connect(
            lambda: run_in_thread(self.start_Zstack_scanning)
        )
        Zstack_Layout.addWidget(self.startButton_stack_scanning, 7, 0)

        self.stopButton_stack_scanning = StylishQT.stop_deleteButton()
        self.stopButton_stack_scanning.setFixedHeight(32)
        self.stopButton_stack_scanning.clicked.connect(
            lambda: self.buttonenabled("stackscan", "stop")
        )
        self.stopButton_stack_scanning.clicked.connect(
            lambda: run_in_thread(self.stop_Zstack_scanning)
        )
        self.stopButton_stack_scanning.setEnabled(False)
        Zstack_Layout.addWidget(self.stopButton_stack_scanning, 7, 1)

        Zstack_widget.setLayout(Zstack_Layout)

        self.scanning_tabs.addTab(Continuous_widget, "Continuous scanning")
        self.scanning_tabs.addTab(Zstack_widget, "Stack scanning")

        # ---------------------------Set tab1 layout---------------------------
        #        pmtmaster = QGridLayout()
        self.layout.addWidget(pmtimageContainer, 0, 0, 3, 1)
        self.layout.addWidget(pmtroiContainer, 1, 1)
        self.layout.addWidget(pmtContourContainer, 2, 1)
        self.layout.addWidget(self.scanning_tabs, 0, 1)

    #        self.layout.setLayout(pmtmaster)

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # --------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------Functions for TAB 'PMT'---------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------
    # **************************************************************************************************************************************
    def generate_poly_roi(self, event):
        """
        For each click event, add a handle to the poly roi

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if not self.flag_is_drawing:
            return
        
        x = int(event.pos().x())
        y = int(event.pos().y())

        qpoint_viewbox = self.pmtvb.mapSceneToView(QPoint(x, y))
        # Get the position of the curser
        point = [qpoint_viewbox.x(), qpoint_viewbox.y()]
        
        self.clicked_points_list.append(point)
        
        # if len(self.clicked_points_list) == 1:
        #     self.click_poly_roi = pg.ROI(point)
        #     self.pmtvb.addItem(self.click_poly_roi)
        # else:
        #     self.click_poly_roi.addFreeHandle(point)
            
        if len(self.clicked_points_list) == 1:
            # In case of first click
            self.starting_point = self.clicked_points_list[0]
            self.starting_point_handle_position = [x, y]
            
        elif len(self.clicked_points_list) == 2:
            
            self.click_poly_roi = pg.PolyLineROI(
                positions=[self.starting_point, point]
            )
            
            self.click_poly_roi.sigHoverEvent.connect(
                lambda: self.show_handle_num(self.click_poly_roi)
            )  # update handle numbers
            
            # self.click_poly_roi.setPen(self.pen)
            self.pmtvb.addItem(self.click_poly_roi)
            self.new_roi = False        
            
            
        else:
            self.click_poly_roi.addFreeHandle(point)

            # Remove closing segment of previous mouse movement
            if len(self.click_poly_roi.segments) > 1:
                self.click_poly_roi.removeSegment(self.click_poly_roi.segments[-1])

            self.click_poly_roi.addSegment(
                self.click_poly_roi.handles[-1]["item"],
                self.click_poly_roi.handles[-2]["item"],
            )

            # Add new closing segment
            self.click_poly_roi.addSegment(
                self.click_poly_roi.handles[0]["item"],
                self.click_poly_roi.handles[-1]["item"],
            )
            
    def keyPressEvent(self, event):
        # print(event.key())
        if event.key() == 70: # If the 'f' key is pressed
            if self.flag_is_drawing:
                self.flag_is_drawing = False
            else:
                self.flag_is_drawing = True
                self.new_roi = True
            
    def buttonenabled(self, button, switch):

        if button == "rasterscan":
            if switch == "start":
                self.startButton_pmt.setEnabled(False)
                self.stopButton.setEnabled(True)

            elif switch == "stop":
                self.startButton_pmt.setEnabled(True)
                self.stopButton.setEnabled(False)

        elif button == "contourscan":
            if switch == "start":  # disable start button and enable stop button
                self.do_contour_sacn.setEnabled(False)
                self.stopButton_contour.setEnabled(True)
            elif switch == "stop":
                self.do_contour_sacn.setEnabled(True)
                self.stopButton_contour.setEnabled(False)

        elif button == "stackscan":
            if switch == "start":
                self.startButton_stack_scanning.setEnabled(False)
                self.stopButton_stack_scanning.setEnabled(True)

            elif switch == "stop":
                self.startButton_stack_scanning.setEnabled(True)
                self.stopButton_stack_scanning.setEnabled(False)

    def measure_pmt(self):
        """
        Do raster scan and update the graph.

        Returns
        -------
        None.

        """
        try:
            self.Daq_sample_rate_pmt = int(self.continuous_scanning_sr_spinbox.value())
    
            # Voltage settings, by default it's equal range square.
            self.Value_voltXMax = self.continuous_scanning_Vrange_spinbox.value()
            self.Value_voltXMin = self.Value_voltXMax * -1
            Value_voltYMin = self.Value_voltXMin
            Value_voltYMax = self.Value_voltXMax
    
            self.Value_xPixels = int(self.Scanning_pixel_num_combobox.value())
            Value_yPixels = self.Value_xPixels
            self.averagenum = int(self.continuous_scanning_average_spinbox.value())
    
            Totalscansamples = self.pmtTest.setWave(
                self.Daq_sample_rate_pmt,
                self.Value_voltXMin,
                self.Value_voltXMax,
                Value_voltYMin,
                Value_voltYMax,
                self.Value_xPixels,
                Value_yPixels,
                self.averagenum,
            )
            time_per_frame_pmt = Totalscansamples / self.Daq_sample_rate_pmt
    
            ScanArrayXnum = int((Totalscansamples / self.averagenum) / Value_yPixels)
    
            # r1 = QRectF(500, 500, ScanArrayXnum, int(Value_yPixels))
            # self.pmtimageitem.setRect(r1)
    
            self.pmtTest.pmtimagingThread.measurement.connect(
                self.update_pmt_Graphs
            )  # Connecting to the measurement signal
            self.pmt_fps_Label.setText("Per frame:  %.4f s" % time_per_frame_pmt)
            self.pmtTest.start()
            
        except:
            print("NI-Daq not connected.")
            self.update_pmt_Graphs(data = np.zeros((Value_yPixels, Value_yPixels)))

    def measure_pmt_contourscan(self):
        
        self.Daq_sample_rate_pmt = int(self.contour_samprate.value())
        
        self.pmtTest_contour.setWave_contourscan(
            self.Daq_sample_rate_pmt,
            self.final_stacked_voltage_signals,
            self.points_per_round,
        )
        contour_freq = self.Daq_sample_rate_pmt / self.points_per_round

        # r1 = QRectF(500, 500, ScanArrayXnum, int(Value_yPixels))
        # self.pmtimageitem.setRect(r1)

        # self.pmtTest_contour.pmtimagingThread_contour.measurement.connect(self.update_pmt_Graphs) #Connecting to the measurement signal
        self.pmt_fps_Label.setText("Contour frequency:  %.4f Hz" % contour_freq)
        self.pmtTest_contour.start()
        self.MessageToMainGUI("---!! Continuous contour scanning !!---" + "\n")

    def saveimage_pmt(self):
        Localimg = Image.fromarray(self.data_pmtcontineous)  # generate an image object
        Localimg.save(
            os.path.join(
                self.savedirectory,
                "PMT_"
                + self.prefixtextboxtext
                + "_"
                + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                + ".tif",
            )
        )  # save as tif
        # np.save(os.path.join(self.savedirectory, 'PMT'+ self.saving_prefix +datetime.now().strftime('%Y-%m-%d_%H-%M-%S')), self.data_pmtcontineous)

    def update_pmt_Graphs(self, data):
        """Update graphs."""

        self.data_pmtcontineous = data
        self.pmtvideoWidget.setImage(data)
        # self.pmtimgroi.setImage(
        #     self.roi.getArrayRegion(data, self.pmtimageitem), levels=(0, data.max())
        # )
        #

        # self.pmtvideoWidget.update_pmt_Window(self.data_pmtcontineous)

    def show_handle_num(self, roi_item):
        """
        Show the number of handles.

        Returns
        -------
        None.

        """
        self.ROIhandles = roi_item.getHandles()
        self.ROIhandles_nubmer = len(self.ROIhandles)
        self.pmt_handlenum_Label.setText("Handle number: %.d" % self.ROIhandles_nubmer)
        
    def regenerate_roi_handles(self):
        """
        Regenerate the handles from desired roi in sequence.

        Returns
        -------
        None.

        """
        current_roi_handles_list = self.contour_ROI_handles_dict[
            "handles_{}".format(self.roi_index_spinbox.value())
        ]
        
        self.pmtvb.removeItem(self.click_poly_roi)
        
        self.click_poly_roi = pg.PolyLineROI(
            current_roi_handles_list, pen=self.ROIpen, closed = True
        )  # , maxBounds=r1
        # self.roi.addRotateHandle([40,0], [0.5, 0.5])
        self.click_poly_roi.sigHoverEvent.connect(
            lambda: self.show_handle_num(self.click_poly_roi)
        )  # update handle numbers

        self.pmtvb.addItem(self.click_poly_roi)  # add ROIs to main image

    def add_coordinates_to_list(self):
        """
        Add one coordinate signals to the loop.

        Returns
        -------
        None.

        """
        # Generate the voltage signals
        self.current_stacked_voltage_signals = self.generate_contour_coordinates(self.click_poly_roi)

        # Place the signals to the corresponding dictionary position
        self.contour_ROI_signals_dict[
            "roi_{}".format(self.roi_index_spinbox.value())
        ] = self.current_stacked_voltage_signals
        
        # Record roi handle positions
        roi_handles_scene_list = []
        
        # From QPoint to list
        for each_item in self.handle_local_coordinate_position_raw_list:
            roi_handles_scene_list.append([each_item[1].x(), each_item[1].y()])

        self.contour_ROI_handles_dict[
            "handles_{}".format(self.roi_index_spinbox.value())
        ] = roi_handles_scene_list

        
    def del_coordinates_from_list(self):
        """
        Remove the last mask from the list.
        """
        del self.contour_ROI_signals_dict[
            "roi_{}".format(self.roi_index_spinbox.value())
        ]       
        
        del self.contour_ROI_handles_dict[
            "handles_{}".format(self.roi_index_spinbox.value())
        ]  

    def generate_final_contour_signals(self):
        """
        Add together all the signals and emit it to other widgets.

        Returns
        -------
        None.

        """
        
        if len(self.contour_ROI_signals_dict) == 1:
            # With only one roi in list
            self.final_stacked_voltage_signals = self.contour_ROI_signals_dict["roi_1"]
        else:
            # With multiple roi added
            temp_list = []
            for each_roi_coordinate in self.contour_ROI_signals_dict:
                temp_list.append(self.contour_ROI_signals_dict[each_roi_coordinate])
            
            self.final_stacked_voltage_signals = np.concatenate(temp_list, axis = 1)
            
        # Number of points in single round of contour scan
        self.points_per_round = len(self.final_stacked_voltage_signals[0])
        
        # print(self.final_stacked_voltage_signals)
        
        # To the main widget Fiumicino
        self.emit_contour_signal()
        
    def go_to_first_point(self):
        """
        Before executing contour scanning, preset galvo positions to first point.

        Returns
        -------
        None.

        """
        first_point_x = self.final_stacked_voltage_signals[:,0][0]
        first_point_y = self.final_stacked_voltage_signals[:,0][1]
    
        print("galvo move to: {}, {}".format(first_point_x, first_point_y))
    
        daq = DAQmission()
        daq.sendSingleAnalog("galvosx", first_point_x)
        daq.sendSingleAnalog("galvosy", first_point_y)
        
    
    def reset_roi_handles(self):
        """
        Reset_roi_handles positions.

        Returns
        -------
        None.

        """
        # self.pmtvb.removeItem(self.roi)
        
        # self.ROIpen = QPen()  # creates a default pen
        # self.ROIpen.setStyle(Qt.DashDotLine)
        # self.ROIpen.setWidth(1)
        # self.ROIpen.setBrush(QColor(0, 161, 255))
        # self.roi = pg.PolyLineROI(
        #     [[0, 0], [80, 0], [80, 80], [0, 80]], closed=True, pen=self.ROIpen
        # )  # , maxBounds=r1
        # # self.roi.addRotateHandle([40,0], [0.5, 0.5])
        # self.roi.sigHoverEvent.connect(
        #     lambda: self.show_handle_num()
        # )  # update handle numbers

        # self.pmtvb.addItem(self.roi)  # add ROIs to main image
        
        #=====================================================================
        try:
            self.pmtvb.removeItem(self.click_poly_roi)
        except:
            pass
        
        self.clicked_points_list = []
        
    def reset_coordinates_dict(self):
    
        self.final_stacked_voltage_signals = None
        self.contour_ROI_signals_dict = {}
        # self.contour_ROI_handles_dict = {}
        
        self.reset_roi_handles()
        
    def generate_contour_coordinates(self, roi_item):
        """
        Geneate the voltage signals according to current ROI's handle positions.

        Returns
        -------
        TYPE
            np.array. (2,n), two rows stack together.

        getSceneHandlePositions IS THE FUNCTION TO GRAP COORDINATES FROM IMAGEITEM REGARDLESS OF IMAGEITEM ZOOMING OR PANNING!!!
        """
        
        self.Daq_sample_rate_pmt = int(self.contour_samprate.value())
        
        self.ROIhandles = roi_item.getHandles()
        self.ROIhandles_nubmer = len(self.ROIhandles)
        contour_point_number = int(self.pointsinContour.value())
        
        # Get the handle positions in the imageitem coordinates
        self.handle_scene_coordinate_position_raw_list = (
            roi_item.getSceneHandlePositions()
        )
        # print(self.handle_scene_coordinate_position_raw_list)
        # # ===== The first placed handle is at the end, put back to front.=====
        # first_placed_handle = self.handle_scene_coordinate_position_raw_list[-1]

        # self.handle_scene_coordinate_position_raw_list.insert(0,first_placed_handle)
        # self.handle_scene_coordinate_position_raw_list.pop(-1)
        

        self.handle_local_coordinate_position_raw_list = (
            roi_item.getLocalHandlePositions()
        )
        
        # put scene positions into numpy array
        self.handle_scene_coordinate_position_array = np.zeros(
            (self.ROIhandles_nubmer, 2)
        )  # n rows, 2 columns
        for i in range(self.ROIhandles_nubmer):
            self.handle_scene_coordinate_position_array[i] = np.array(
                [
                    self.handle_scene_coordinate_position_raw_list[i][1].x(),
                    self.handle_scene_coordinate_position_raw_list[i][1].y(),
                ]
            )
        print(self.handle_scene_coordinate_position_array)
        if self.contour_strategy.currentText() == "Evenly between":
            # Interpolation
            self.point_num_per_line = int(
                contour_point_number / self.ROIhandles_nubmer
            )
            self.Interpolation_number = self.point_num_per_line - 1
            
            # ====== Doing the uniform interpolation in between handles =======
            self.handle_scene_coordinate_position_array_expanded = \
                self.interpolate_evenly_between_nodes(node_number = self.ROIhandles_nubmer,
                                                  point_num_per_line = self.point_num_per_line,
                                                  node_position_array = self.handle_scene_coordinate_position_array)
            #=================================================================
            
            self.handle_viewbox_coordinate_position_array_expanded = np.zeros(
                (contour_point_number, 2)
            )  # n rows, 2 columns
            # Maps from scene coordinates to the coordinate system displayed inside the ViewBox
            for i in range(contour_point_number):
                qpoint_Scene = QPoint(
                    int(self.handle_scene_coordinate_position_array_expanded[i][0]),
                    int(self.handle_scene_coordinate_position_array_expanded[i][1]),
                )
                qpoint_viewbox = self.pmtvb.mapSceneToView(qpoint_Scene)
                self.handle_viewbox_coordinate_position_array_expanded[i] = np.array(
                    [qpoint_viewbox.x(), qpoint_viewbox.y()]
                )

            """Transform into Voltages to galvos"""
            """coordinates in the view box(handle_viewbox_coordinate_position_array_expanded_x) are equivalent to voltages sending out"""
            
            self.handle_viewbox_coordinate_position_array_expanded_x, \
            self.handle_viewbox_coordinate_position_array_expanded_y = \
            self.convert_coordinates_to_voltage(Value_xPixels = self.Value_xPixels, Value_voltXMax = self.Value_voltXMax,
                                                contour_point_number = contour_point_number, 
                                                handle_viewbox_coordinates = self.handle_viewbox_coordinate_position_array_expanded)
                    
            # ================= The signals to NIDAQ ==================
            current_stacked_voltage_signals = np.vstack(
                (
                    self.handle_viewbox_coordinate_position_array_expanded_x,
                    self.handle_viewbox_coordinate_position_array_expanded_y,
                )
            )
                    
            #================= Speed and acceleration check ===================
            self.speed_acceleration_check(self.Daq_sample_rate_pmt,
                                          self.handle_viewbox_coordinate_position_array_expanded_x,
                                          self.handle_viewbox_coordinate_position_array_expanded_y)
            

        #============================ Uniform =================================
        
        if self.contour_strategy.currentText() == "Uniform":
            # Calculate the total distance
            self.total_distance = 0
            for i in range(self.ROIhandles_nubmer):
                if i != (self.ROIhandles_nubmer - 1):
                    Interpolation_x_diff = (
                        self.handle_scene_coordinate_position_array[i + 1][0]
                        - self.handle_scene_coordinate_position_array[i][0]
                    )
                    Interpolation_y_diff = (
                        self.handle_scene_coordinate_position_array[i + 1][1]
                        - self.handle_scene_coordinate_position_array[i][1]
                    )
                    distance_vector = (
                        Interpolation_x_diff ** 2 + Interpolation_y_diff ** 2
                    ) ** (0.5)
                    self.total_distance = self.total_distance + distance_vector
                else:
                    Interpolation_x_diff = (
                        self.handle_scene_coordinate_position_array[0][0]
                        - self.handle_scene_coordinate_position_array[-1][0]
                    )
                    Interpolation_y_diff = (
                        self.handle_scene_coordinate_position_array[0][1]
                        - self.handle_scene_coordinate_position_array[-1][1]
                    )
                    distance_vector = (
                        Interpolation_x_diff ** 2 + Interpolation_y_diff ** 2
                    ) ** (0.5)
                    self.total_distance = self.total_distance + distance_vector

            self.averaged_uniform_step = self.total_distance / contour_point_number

            print(self.averaged_uniform_step)
            print(self.handle_scene_coordinate_position_array)

            for i in range(self.ROIhandles_nubmer):
                if i == 0:
                    Interpolation_x_diff = (
                        self.handle_scene_coordinate_position_array[i + 1][0]
                        - self.handle_scene_coordinate_position_array[i][0]
                    )
                    Interpolation_y_diff = (
                        self.handle_scene_coordinate_position_array[i + 1][1]
                        - self.handle_scene_coordinate_position_array[i][1]
                    )
                    distance_vector = (
                        Interpolation_x_diff ** 2 + Interpolation_y_diff ** 2
                    ) ** (0.5)
                    num_of_Interpolation = distance_vector // self.averaged_uniform_step

                    # Interpolation_remaining = distance_vector%self.averaged_uniform_step
                    self.Interpolation_remaining_fornextround = (
                        self.averaged_uniform_step
                        * (
                            1
                            - (
                                distance_vector / self.averaged_uniform_step
                                - num_of_Interpolation
                            )
                        )
                    )
                    print(
                        "Interpolation_remaining_fornextround: "
                        + str(self.Interpolation_remaining_fornextround)
                    )
                    self.Interpolation_x_step = Interpolation_x_diff / (
                        distance_vector / self.averaged_uniform_step
                    )
                    self.Interpolation_y_step = Interpolation_y_diff / (
                        distance_vector / self.averaged_uniform_step
                    )

                    Interpolation_temp = np.array(
                        [
                            [
                                self.handle_scene_coordinate_position_array[i][0],
                                self.handle_scene_coordinate_position_array[i][1],
                            ],
                            [
                                self.handle_scene_coordinate_position_array[i + 1][0],
                                self.handle_scene_coordinate_position_array[i + 1][1],
                            ],
                        ]
                    )

                    for j in range(int(num_of_Interpolation)):
                        Interpolation_temp = np.insert(
                            Interpolation_temp,
                            -1,
                            [
                                self.handle_scene_coordinate_position_array[i][0]
                                + (j + 1) * self.Interpolation_x_step,
                                self.handle_scene_coordinate_position_array[i + 1][1]
                                + (j + 1) * self.Interpolation_y_step,
                            ],
                            axis=0,
                        )
                    Interpolation_temp = np.delete(Interpolation_temp, -1, axis=0)

                    self.handle_scene_coordinate_position_array_expanded_uniform = (
                        Interpolation_temp
                    )

                elif i != (self.ROIhandles_nubmer - 1):
                    Interpolation_x_diff = (
                        self.handle_scene_coordinate_position_array[i + 1][0]
                        - self.handle_scene_coordinate_position_array[i][0]
                    )
                    Interpolation_y_diff = (
                        self.handle_scene_coordinate_position_array[i + 1][1]
                        - self.handle_scene_coordinate_position_array[i][1]
                    )
                    distance_vector = (
                        Interpolation_x_diff ** 2 + Interpolation_y_diff ** 2
                    ) ** (0.5)
                    num_of_Interpolation = (
                        distance_vector - self.Interpolation_remaining_fornextround
                    ) // self.averaged_uniform_step
                    print(
                        "Interpolation_remaining_fornextround: "
                        + str(self.Interpolation_remaining_fornextround)
                    )

                    if self.Interpolation_remaining_fornextround != 0:
                        self.Interpolation_remaining_fornextround_x = Interpolation_x_diff / (
                            distance_vector / self.Interpolation_remaining_fornextround
                        )  # (self.Interpolation_remaining_fornextround/distance_vector)*Interpolation_x_diff
                        self.Interpolation_remaining_fornextround_y = Interpolation_y_diff / (
                            distance_vector / self.Interpolation_remaining_fornextround
                        )  # (self.Interpolation_remaining_fornextround/distance_vector)*Interpolation_y_diff
                    else:
                        self.Interpolation_remaining_fornextround_x = 0
                        self.Interpolation_remaining_fornextround_y = 0

                    # Reset the starting point
                    Interpolation_x_diff = (
                        self.handle_scene_coordinate_position_array[i + 1][0]
                        - self.handle_scene_coordinate_position_array[i][0]
                        - self.Interpolation_remaining_fornextround_x
                    )
                    Interpolation_y_diff = (
                        self.handle_scene_coordinate_position_array[i + 1][1]
                        - self.handle_scene_coordinate_position_array[i][1]
                        - self.Interpolation_remaining_fornextround_y
                    )

                    self.Interpolation_x_step = Interpolation_x_diff / (
                        (distance_vector - self.Interpolation_remaining_fornextround)
                        / self.averaged_uniform_step
                    )
                    self.Interpolation_y_step = Interpolation_y_diff / (
                        (distance_vector - self.Interpolation_remaining_fornextround)
                        / self.averaged_uniform_step
                    )

                    Interpolation_temp = np.array(
                        [
                            [
                                self.handle_scene_coordinate_position_array[i][0]
                                + self.Interpolation_remaining_fornextround_x,
                                self.handle_scene_coordinate_position_array[i][1]
                                + self.Interpolation_remaining_fornextround_y,
                            ],
                            [
                                self.handle_scene_coordinate_position_array[i + 1][0],
                                self.handle_scene_coordinate_position_array[i + 1][1],
                            ],
                        ]
                    )

                    for j in range(int(num_of_Interpolation)):
                        Interpolation_temp = np.insert(
                            Interpolation_temp,
                            -1,
                            [
                                self.handle_scene_coordinate_position_array[i][0]
                                + self.Interpolation_remaining_fornextround_x
                                + (j + 1) * self.Interpolation_x_step,
                                self.handle_scene_coordinate_position_array[i][1]
                                + self.Interpolation_remaining_fornextround_y
                                + (j + 1) * self.Interpolation_y_step,
                            ],
                            axis=0,
                        )
                    Interpolation_temp = np.delete(Interpolation_temp, -1, axis=0)

                    self.handle_scene_coordinate_position_array_expanded_uniform = np.append(
                        self.handle_scene_coordinate_position_array_expanded_uniform,
                        Interpolation_temp,
                        axis=0,
                    )

                    self.Interpolation_remaining_fornextround = (
                        self.averaged_uniform_step
                        * (
                            1
                            - (
                                (
                                    distance_vector
                                    - self.Interpolation_remaining_fornextround
                                )
                                / self.averaged_uniform_step
                                - num_of_Interpolation
                            )
                        )
                    )

                else:  # connect the first and the last
                    Interpolation_x_diff = (
                        self.handle_scene_coordinate_position_array[0][0]
                        - self.handle_scene_coordinate_position_array[-1][0]
                    )
                    Interpolation_y_diff = (
                        self.handle_scene_coordinate_position_array[0][1]
                        - self.handle_scene_coordinate_position_array[-1][1]
                    )
                    distance_vector = (
                        Interpolation_x_diff ** 2 + Interpolation_y_diff ** 2
                    ) ** (0.5)
                    num_of_Interpolation = (
                        distance_vector - self.Interpolation_remaining_fornextround
                    ) // self.averaged_uniform_step

                    # self.Interpolation_remaining_fornextround = self.averaged_uniform_step*(1-((distance_vector-self.Interpolation_remaining_fornextround)/self.averaged_uniform_step-num_of_Interpolation))
                    self.Interpolation_remaining_fornextround_x = (
                        self.Interpolation_remaining_fornextround / distance_vector
                    ) * Interpolation_x_diff
                    self.Interpolation_remaining_fornextround_y = (
                        self.Interpolation_remaining_fornextround / distance_vector
                    ) * Interpolation_y_diff

                    # Reset the starting point
                    Interpolation_x_diff = (
                        self.handle_scene_coordinate_position_array[0][0]
                        - self.handle_scene_coordinate_position_array[i][0]
                        + self.Interpolation_remaining_fornextround_x
                    )
                    Interpolation_y_diff = (
                        self.handle_scene_coordinate_position_array[0][1]
                        - self.handle_scene_coordinate_position_array[i][1]
                        + self.Interpolation_remaining_fornextround_y
                    )

                    self.Interpolation_x_step = Interpolation_x_diff / (
                        (distance_vector - self.Interpolation_remaining_fornextround)
                        / self.averaged_uniform_step
                    )
                    self.Interpolation_y_step = Interpolation_y_diff / (
                        (distance_vector - self.Interpolation_remaining_fornextround)
                        / self.averaged_uniform_step
                    )

                    Interpolation_temp = np.array(
                        [
                            [
                                self.handle_scene_coordinate_position_array[-1][0]
                                + self.Interpolation_remaining_fornextround_x,
                                self.handle_scene_coordinate_position_array[-1][1]
                                + self.Interpolation_remaining_fornextround_y,
                            ],
                            [
                                self.handle_scene_coordinate_position_array[0][0],
                                self.handle_scene_coordinate_position_array[0][1],
                            ],
                        ]
                    )

                    for j in range(int(num_of_Interpolation)):
                        Interpolation_temp = np.insert(
                            Interpolation_temp,
                            -1,
                            [
                                self.handle_scene_coordinate_position_array[-1][0]
                                + self.Interpolation_remaining_fornextround_x
                                + (j + 1) * self.Interpolation_x_step,
                                self.handle_scene_coordinate_position_array[-1][1]
                                + self.Interpolation_remaining_fornextround_y
                                + (j + 1) * self.Interpolation_y_step,
                            ],
                            axis=0,
                        )
                    Interpolation_temp = np.delete(Interpolation_temp, -1, axis=0)

                    self.handle_scene_coordinate_position_array_expanded_uniform = np.append(
                        self.handle_scene_coordinate_position_array_expanded_uniform,
                        Interpolation_temp,
                        axis=0,
                    )

            print(self.handle_scene_coordinate_position_array_expanded_uniform)
            print(self.handle_scene_coordinate_position_array_expanded_uniform.shape)
            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            self.handle_viewbox_coordinate_position_array_expanded = np.zeros(
                (contour_point_number, 2)
            )  # n rows, 2 columns
            # Maps from scene coordinates to the coordinate system displayed inside the ViewBox
            for i in range(contour_point_number):
                qpoint_Scene = QPoint(
                    self.handle_scene_coordinate_position_array_expanded_uniform[i][0],
                    self.handle_scene_coordinate_position_array_expanded_uniform[i][1],
                )
                qpoint_viewbox = self.pmtvb.mapSceneToView(qpoint_Scene)
                self.handle_viewbox_coordinate_position_array_expanded[i] = np.array(
                    [qpoint_viewbox.x(), qpoint_viewbox.y()]
                )


            """Transform into Voltages to galvos"""

            self.handle_viewbox_coordinate_position_array_expanded_x,
            self.handle_viewbox_coordinate_position_array_expanded_y = \
            self.convert_coordinates_to_voltage(Value_xPixels = self.Value_xPixels, Value_voltXMax = self.Value_voltXMax,
                                                contour_point_number = contour_point_number, 
                                                handle_viewbox_coordinates = self.handle_viewbox_coordinate_position_array_expanded)
                    
            # ================= The signals to NIDAQ ==================
            current_stacked_voltage_signals = np.vstack(
                (
                    self.handle_viewbox_coordinate_position_array_expanded_x,
                    self.handle_viewbox_coordinate_position_array_expanded_y,
                )
            )
                    
                    
            #================= Speed and acceleration check ===================
            self.speed_acceleration_check(self.Daq_sample_rate_pmt,
                                          self.handle_viewbox_coordinate_position_array_expanded_x,
                                          self.handle_viewbox_coordinate_position_array_expanded_y)
        
        # print(current_stacked_voltage_signals)
        
        # stacked_voltage_signals_length_hori = len(current_stacked_voltage_signals[1])
        
        # try:
        #     # Setting the starting point
        #     # starting_point_index = np.argmax(current_stacked_voltage_signals[1,:]) # Set lowest point in view as start 

        #     starting_point_index = np.where(np.all(current_stacked_voltage_signals == np.array(self.starting_point), axis = 0))[0][0]
            
        #     # Set two parts
        #     moving_forward_part = current_stacked_voltage_signals[:,\
        #                                                           starting_point_index:stacked_voltage_signals_length_hori]
        #     moving_backward_part = current_stacked_voltage_signals[:,\
        #                                                           0:starting_point_index]
        #     # Create container
        #     resequenced_stacked_voltage_signals = np.zeros((current_stacked_voltage_signals.shape[0],\
        #                                                     current_stacked_voltage_signals.shape[1]))
        #     # Fill in container with first part and 2nd part
        #     resequenced_stacked_voltage_signals[:,\
        #                                         0:len(moving_forward_part[1])] \
        #         = moving_forward_part
            
        #     resequenced_stacked_voltage_signals[:,\
        #                                         len(moving_forward_part[1]):stacked_voltage_signals_length_hori] \
        #         = moving_backward_part 
        # except:
        #     print("Fail to find starting point.")
        #     resequenced_stacked_voltage_signals = current_stacked_voltage_signals
        
        resequenced_stacked_voltage_signals = current_stacked_voltage_signals
        print(resequenced_stacked_voltage_signals)

        return resequenced_stacked_voltage_signals
    
    def interpolate_evenly_between_nodes(self, node_number, point_num_per_line, node_position_array):
        """
        Interpolate evenly in between roi handles

        Parameters
        ----------
        node_number : int
            Number of handles in roi.
        point_num_per_line : int
            Number of points per line desired.
        node_position_array : np.array
            DESCRIPTION.

        Returns
        -------
        interpolated_array. (n,2), 2 columns

        """
        # --------------------------------------Interpolation from first to last---------------------------------------------
        for i in range(node_number - 1):
            Interpolation_x_diff = (
                node_position_array[i + 1][0]
                - node_position_array[i][0]
            )
            Interpolation_y_diff = (
                node_position_array[i + 1][1]
                - node_position_array[i][1]
            )

            Interpolation_x_step = (
                Interpolation_x_diff / point_num_per_line
            )
            Interpolation_y_step = (
                Interpolation_y_diff / point_num_per_line
            )

            Interpolation_temp = np.array(
                [
                    [
                        node_position_array[i][0],
                        node_position_array[i][1],
                    ],
                    [
                        node_position_array[i + 1][0],
                        node_position_array[i + 1][1],
                    ],
                ]
            )

            for j in range(point_num_per_line - 1):
                Interpolation_temp = np.insert(
                    Interpolation_temp,
                    1,
                    [
                        node_position_array[i + 1][0]
                        - (j + 1) * Interpolation_x_step,
                        node_position_array[i + 1][1]
                        - (j + 1) * Interpolation_y_step,
                    ],
                    axis=0,
                )
            Interpolation_temp = np.delete(Interpolation_temp, 0, 0)
            if i == 0:
                interpolated_array = (
                    Interpolation_temp
                )
            else:
                interpolated_array = np.append(
                    interpolated_array,
                    Interpolation_temp,
                    axis=0,
                )
                # self.handle_scene_coordinate_position_array_expanded=np.delete(self.handle_scene_coordinate_position_array_expanded, 0, 0)

        # Interpolation between last and first
        Interpolation_x_diff = (
            node_position_array[0][0]
            - node_position_array[-1][0]
        )
        Interpolation_y_diff = (
            node_position_array[0][1]
            - node_position_array[-1][1]
        )

        Interpolation_x_step = (
            Interpolation_x_diff / point_num_per_line
        )
        Interpolation_y_step = (
            Interpolation_y_diff / point_num_per_line
        )

        Interpolation_temp = np.array(
            [
                [
                    node_position_array[-1][0],
                    node_position_array[-1][1],
                ],
                [
                    node_position_array[0][0],
                    node_position_array[0][1],
                ],
            ]
        )

        for j in range(point_num_per_line - 1):
            Interpolation_temp = np.insert(
                Interpolation_temp,
                1,
                [
                    node_position_array[0][0]
                    - (j + 1) * Interpolation_x_step,
                    node_position_array[0][1]
                    - (j + 1) * Interpolation_y_step,
                ],
                axis=0,
            )
        Interpolation_temp = np.delete(Interpolation_temp, 0, 0)
        # Interpolation_temp = np.flip(Interpolation_temp, 0)

        interpolated_array = np.append(
            interpolated_array,
            Interpolation_temp,
            axis=0,
        )
        
        # ===== The first placed handle is at the end, put back to front.=====

        interpolated_array_modified = np.zeros([interpolated_array.shape[0], interpolated_array.shape[1]])
        interpolated_array_modified[0,:] = interpolated_array[-1,:]
        interpolated_array_modified[1:interpolated_array.shape[0],:] = interpolated_array[0:interpolated_array.shape[0] -1, :]

        return interpolated_array_modified
    
    def convert_coordinates_to_voltage(self, Value_xPixels, Value_voltXMax, 
                                       contour_point_number, 
                                       handle_viewbox_coordinates):
        """
        Transform the viewbox coordinates to galvo scanning voltage signals

        Parameters
        ----------
        Value_xPixels : int
            pixel number in the image.
        Value_voltXMax : int
            Galvo scanning voltage.
        contour_point_number : int
            Number of points in one contour scan signal.
        handle_viewbox_coordinates : np.array, (n,2)
            DESCRIPTION.

        Returns
        -------
        transformed_x : TYPE
            DESCRIPTION.
        transformed_y : TYPE
            DESCRIPTION.

        """
        if Value_xPixels == 500:
            if Value_voltXMax == 3:
                # for 500 x axis, the real ramp region sits around 52~552 out of 0~758
                handle_viewbox_coordinates[:, 0] = (
                    (handle_viewbox_coordinates[:, 0])
                    / 500
                ) * 6 - 3  # (handle_viewbox_coordinates[:,0]-constants.pmt_3v_indentation_pixels)
                handle_viewbox_coordinates[:, 1] = (
                    (handle_viewbox_coordinates[:, 1])
                    / 500
                ) * 6 - 3
                handle_viewbox_coordinates = np.around(
                    handle_viewbox_coordinates,
                    decimals=3,
                )
                # shape into (n,) and stack
                transformed_x = (
                    np.resize(
                        handle_viewbox_coordinates[
                            :, 0
                        ],
                        (contour_point_number,),
                    )
                )
                transformed_y = (
                    np.resize(
                        handle_viewbox_coordinates[
                            :, 1
                        ],
                        (contour_point_number,),
                    )
                )
                    
        return transformed_x, transformed_y
    
    def speed_acceleration_check(self, sampling_rate, trace_x, trace_y):
        """
        Check the speed and acceleration of galvos

        Parameters
        ----------
        sampling_rate : int
            DESCRIPTION.
        trace_x : np.array
            DESCRIPTION.
        trace_y : np.array
            DESCRIPTION.

        Returns
        -------
        None.

        """
        time_gap = 1 / sampling_rate
        contour_x_speed = (
            np.diff(trace_x)
            / time_gap
        )
        contour_y_speed = (
            np.diff(trace_y)
            / time_gap
        )

        contour_x_acceleration = np.diff(contour_x_speed) / time_gap
        contour_y_acceleration = np.diff(contour_y_speed) / time_gap

        constants = HardwareConstants()
        speedGalvo = constants.maxGalvoSpeed  # Volt/s
        aGalvo = constants.maxGalvoAccel  # Acceleration galvo in volt/s^2
        # print(np.amax(abs(contour_x_speed)))
        # print(np.amax(abs(contour_y_speed)))
        # print(np.amax(abs(contour_x_acceleration)))
        # print(np.amax(abs(contour_y_acceleration)))

        print(
            str(np.mean(abs(contour_x_speed)))
            + " and mean y speed:"
            + str(np.mean(abs(contour_y_speed)))
        )
        print(
            str(np.mean(abs(contour_x_acceleration)))
            + " and mean y acceleration:"
            + str(np.mean(abs(contour_y_acceleration)))
        )

        if speedGalvo > np.amax(abs(contour_x_speed)) and speedGalvo > np.amax(
            abs(contour_y_speed)
        ):
            print("Contour speed is OK")
            self.MessageToMainGUI("Contour speed is OK" + "\n")
        else:
            QMessageBox.warning(self, "OverLoad", "Speed too high!", QMessageBox.Ok)
        if aGalvo > np.amax(abs(contour_x_acceleration)) and aGalvo > np.amax(
            abs(contour_y_acceleration)
        ):
            print("Contour acceleration is OK")
            self.MessageToMainGUI("Contour acceleration is OK" + "\n")
        else:
            QMessageBox.warning(
                self, "OverLoad", "Acceleration too high!", QMessageBox.Ok
            )
    
    def emit_contour_signal(self):
        """
        Emit generated contour signals to the main widget, then pass to waveform widget.

        Returns
        -------
        None.

        """
        
        self.SignalForContourScanning.emit(
            int(self.points_per_round),
            self.Daq_sample_rate_pmt,
            (1 / int(self.contour_samprate.value()) * 1000) * self.points_per_round, # time per contour scan
            self.final_stacked_voltage_signals[0],
            self.final_stacked_voltage_signals[1],
        )
    
    # def generate_contour_for_waveform(self):
    #     self.contour_time = int(self.textbox1L.value())
    #     self.time_per_contour = (
    #         1 / int(self.contour_samprate.value()) * 1000
    #     ) * self.contour_point_number
    #     repeatnum_contour = int(self.contour_time / self.time_per_contour)
    #     self.repeated_contoursamples_1 = np.tile(
    #         self.handle_viewbox_coordinate_position_array_expanded_x, repeatnum_contour
    #     )
    #     self.repeated_contoursamples_2 = np.tile(
    #         self.handle_viewbox_coordinate_position_array_expanded_y, repeatnum_contour
    #     )

    #     self.handle_viewbox_coordinate_position_array_expanded_forDaq_waveform = (
    #         np.vstack((self.repeated_contoursamples_1, self.repeated_contoursamples_2))
    #     )

    #     return self.handle_viewbox_coordinate_position_array_expanded_forDaq_waveform

    # def generate_galvos_contour_graphy(self):

    #     self.xlabelhere_galvos = (
    #         np.arange(
    #             len(
    #                 self.handle_viewbox_coordinate_position_array_expanded_forDaq_waveform[
    #                     1, :
    #                 ]
    #             )
    #         )
    #         / self.Daq_sample_rate_pmt
    #     )
    #     self.PlotDataItem_galvos = PlotDataItem(
    #         self.xlabelhere_galvos,
    #         self.handle_viewbox_coordinate_position_array_expanded_forDaq_waveform[
    #             1, :
    #         ],
    #     )
    #     self.PlotDataItem_galvos.setDownsampling(auto=True, method="mean")
    #     self.PlotDataItem_galvos.setPen("w")

    #     self.pw.addItem(self.PlotDataItem_galvos)
    #     self.textitem_galvos = pg.TextItem(text="Contour", color=("w"), anchor=(1, 1))
    #     self.textitem_galvos.setPos(0, 5)
    #     self.pw.addItem(self.textitem_galvos)


    def start_Zstack_scanning(self):
        """
        Create the stack scanning instance and run.

        Returns
        -------
        None.

        """
        saving_dir = self.savedirectory
        z_depth = self.stack_scanning_depth_spinbox.value()
        z_step_size = self.stack_scanning_stepsize_spinbox.value()
        imaging_conditions = {
            "Daq_sample_rate": self.stack_scanning_sampling_rate_spinbox.value(),
            "edge_volt": self.stack_scanning_Vrange_spinbox.value(),
            "pixel_number": self.stack_scanning_Pnumber_spinbox.value(),
            "average_number": self.stack_scanning_Avgnumber_spinbox.value(),
        }

        self.zstack_ins = PMT_zscan(
            saving_dir, z_depth, z_step_size, imaging_conditions
        )
        self.zstack_ins.start_scan()

    def stop_Zstack_scanning(self):
        self.zstack_ins.stop_scan()

    def MessageToMainGUI(self, text):
        self.MessageBack.emit(text)

    def stopMeasurement_pmt(self):
        """Stop the seal test."""
        self.pmtTest.aboutToQuitHandler()

    def stopMeasurement_pmt_contour(self):
        """Stop the seal test."""
        self.pmtTest_contour.aboutToQuitHandler()
        self.MessageToMainGUI("---!! Contour stopped !!---" + "\n")

    #    def closeEvent(self, event):
    #
    #        QtWidgets.QApplication.quit()
    #        event.accept()
    '''    
    def closeEvent(self, event):
        """On closing the application we have to make sure that the measuremnt
        stops and the device gets freed."""
        self.stopMeasurement()
    '''


if __name__ == "__main__":

    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        pg.setConfigOptions(imageAxisOrder="row-major")
        mainwin = PMTWidgetUI()
        mainwin.show()
        app.exec_()

    run_app()
