# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:27:04 2020

@author: xinmeng

                                Image analysis GUI
"""

import csv
import logging
import math
import os
import re
import sys
import threading
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyqtgraph as pg
import tifffile as skimtiff
from PIL import Image
from PyQt5 import QtWidgets
from PyQt5.QtCore import QRectF, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QWidget,
)
from scipy import signal
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt, savgol_filter
from skimage import io
from tifffile import TiffFile

from .. import StylishQT
from ..NIDAQ import waveform_specification
from . import matig_of_handmatig
from .ImageProcessing import PatchAnalysis, ProcessImage


class AnalysisWidgetUI(QWidget):
    # waveforms_generated = pyqtSignal(object, object, list, int)
    # SignalForContourScanning = pyqtSignal(int, int, int, np.ndarray, np.ndarray)
    MessageBack = pyqtSignal(str)
    Cellselection_DMD_mask_contour = pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFont(QFont("Arial"))

        self.setMinimumSize(1250, 850)
        self.setWindowTitle("AnalysisWidget")
        self.layout = QGridLayout(self)
        self.savedirectory = ""
        self.OC = 0.1  # Patch clamp constant
        # === GUI for Data analysis tab ===
        readimageContainer = QGroupBox("Readin images")
        self.readimageLayout = QGridLayout()

        self.Construct_name = QLineEdit(self)
        self.Construct_name.setPlaceholderText("Enter construct name")
        self.Construct_name.setFixedWidth(150)
        self.readimageLayout.addWidget(self.Construct_name, 1, 0)

        self.switch_Vp_or_camtrace = QComboBox()
        self.switch_Vp_or_camtrace.addItems(
            ["Correlate to Vp", "Correlate to video", "Photocurrent"]
        )
        self.readimageLayout.addWidget(self.switch_Vp_or_camtrace, 1, 1)

        self.textbox_directory_name = QLineEdit(self)
        self.readimageLayout.addWidget(self.textbox_directory_name, 1, 6)

        # self.button_browse = QPushButton('Set data folder', self)
        # self.readimageLayout.addWidget(self.button_browse, 1, 5)

        # self.button_browse.clicked.connect(self.getfile)

        self.button_load = StylishQT.loadButton()
        self.button_load.setFixedWidth(100)
        self.button_load.setToolTip("Choose data folder and load")
        self.readimageLayout.addWidget(self.button_load, 1, 7)
        self.button_load.clicked.connect(self.getfile)

        self.Spincamsamplingrate = QSpinBox(self)
        self.Spincamsamplingrate.setMaximum(20000)
        self.Spincamsamplingrate.setValue(1000)
        self.Spincamsamplingrate.setSingleStep(500)
        self.readimageLayout.addWidget(self.Spincamsamplingrate, 1, 3)
        self.readimageLayout.addWidget(QLabel("Camera FPS:"), 1, 2)

        self.VoltageStepFrequencyQSpinBox = QSpinBox(self)
        self.VoltageStepFrequencyQSpinBox.setMaximum(20000)
        self.VoltageStepFrequencyQSpinBox.setValue(5)
        self.VoltageStepFrequencyQSpinBox.setSingleStep(5)
        self.readimageLayout.addWidget(self.VoltageStepFrequencyQSpinBox, 1, 5)
        self.readimageLayout.addWidget(QLabel("Voltage step frequency:"), 1, 4)

        self.run_analysis_button = StylishQT.runButton()
        self.run_analysis_button.setFixedWidth(100)
        self.run_analysis_button.setEnabled(False)
        self.readimageLayout.addWidget(self.run_analysis_button, 1, 8)

        self.run_analysis_button.clicked.connect(self.finish_analysis)

        self.button_clearpolts = StylishQT.cleanButton()
        self.button_clearpolts.setFixedWidth(100)
        self.readimageLayout.addWidget(self.button_clearpolts, 1, 9)

        self.button_clearpolts.clicked.connect(self.clearplots)

        readimageContainer.setLayout(self.readimageLayout)
        readimageContainer.setMaximumHeight(120)

        # === Image analysis display Tab ===
        Display_Container = QGroupBox("Image analysis display")
        Display_Layout = QGridLayout()
        # Setting tabs
        Display_Container_tabs = QTabWidget()

        # ------------------------------------------------------Image Analysis-Average window-------------------------------------------------------
        image_display_container_layout = QGridLayout()

        imageanalysis_average_Container = QGroupBox("Background selection")
        self.imageanalysisLayout_average = QGridLayout()

        # self.pw_averageimage = averageimagewindow()
        self.pw_averageimage = pg.ImageView()
        self.pw_averageimage.ui.roiBtn.hide()
        self.pw_averageimage.ui.menuBtn.hide()

        self.roi_average = pg.PolyLineROI(
            [[0, 0], [0, 30], [30, 30], [30, 0]], closed=True
        )
        self.roi_average = pg.RectROI(
            [0, 0], [30, 30], centered=True, sideScalers=True
        )
        self.pw_averageimage.view.addItem(self.roi_average)
        # self.pw_weightimage = weightedimagewindow()
        self.imageanalysisLayout_average.addWidget(
            self.pw_averageimage, 0, 0, 5, 3
        )

        imageanalysis_average_Container.setLayout(
            self.imageanalysisLayout_average
        )
        imageanalysis_average_Container.setMinimumHeight(180)
        # ------------------------------------------------------Image Analysis-weighV window-------------------------------------------------------
        imageanalysis_weight_Container = QGroupBox("Weighted image")
        self.imageanalysisLayout_weight = QGridLayout()

        # self.pw_averageimage = averageimagewindow()
        self.pw_weightimage = pg.ImageView()
        self.pw_weightimage.ui.roiBtn.hide()
        self.pw_weightimage.ui.menuBtn.hide()

        self.roi_weighted = pg.PolyLineROI(
            [[0, 0], [0, 30], [30, 30], [30, 0]], closed=True
        )
        self.pw_weightimage.view.addItem(self.roi_weighted)
        # self.pw_weightimage = weightedimagewindow()
        self.imageanalysisLayout_weight.addWidget(
            self.pw_weightimage, 0, 0, 5, 3
        )

        imageanalysis_weight_Container.setLayout(
            self.imageanalysisLayout_weight
        )
        imageanalysis_weight_Container.setMinimumHeight(180)

        image_display_container_layout.addWidget(
            imageanalysis_average_Container, 0, 0
        )
        image_display_container_layout.addWidget(
            imageanalysis_weight_Container, 0, 1
        )

        Display_Container_tabs_tab3 = PlotAnalysisGUI()
        # Display_Container_tabs_tab3.setLayout(self.Curvedisplay_Layout)

        # Display_Container_tabs_tab2 = QWidget()
        # Display_Container_tabs_tab2.setLayout(self.VIdisplay_Layout)

        Display_Container_tabs_Galvo_WidgetInstance = QWidget()
        Display_Container_tabs_Galvo_WidgetInstance.setLayout(
            image_display_container_layout
        )

        # self.Display_Container_tabs_Cellselection = QWidget()
        # self.Display_Container_tabs_Cellselection_layout = QGridLayout()

        # self.show_cellselection_gui_button = QPushButton('show')
        # self.show_cellselection_gui_button.clicked.connect(self.show_cellselection_gui)
        # self.Display_Container_tabs_Cellselection_layout.addWidget(self.show_cellselection_gui_button, 0,0)
        # self.Display_Container_tabs_Cellselection.setLayout(self.Display_Container_tabs_Cellselection_layout)

        # === Show trace ===
        Display_Container_tabs_tab4 = QWidget()
        Display_Container_tabs_tab4_layout = QGridLayout()

        self.waveform_samplingrate_box = QSpinBox(self)
        self.waveform_samplingrate_box.setMaximum(250000)
        self.waveform_samplingrate_box.setValue(50000)
        self.waveform_samplingrate_box.setSingleStep(500)
        Display_Container_tabs_tab4_layout.addWidget(
            self.waveform_samplingrate_box, 0, 1
        )
        Display_Container_tabs_tab4_layout.addWidget(
            QLabel("Sampling rate:"), 0, 0
        )

        self.textbox_single_waveform_filename = QLineEdit(self)
        Display_Container_tabs_tab4_layout.addWidget(
            self.textbox_single_waveform_filename, 0, 2
        )

        self.button_browse_tab4 = QPushButton("Browse", self)
        Display_Container_tabs_tab4_layout.addWidget(
            self.button_browse_tab4, 0, 3
        )

        self.button_browse_tab4.clicked.connect(self.get_single_waveform)

        Display_Container_tabs_tab4.setLayout(
            Display_Container_tabs_tab4_layout
        )

        # - Ian's automation tab -
        background_select_box = QGroupBox("Step 1 - Background selection")
        vbox_background = QGridLayout()

        vbox_background.addWidget(
            QLabel(
                "First select root folder, pathfinding will search each possible directory for .tiff files and show them to you where you can use your mouse to select an area nearby the cell for background \nselection. Minimum box size is 30, when size is reached, program will automatically show you the next cell. After all cells are complete the plots will disappear and a file named \ncell_locations.csv will be saved in the selected directory containing all background selections."
            ),
            0,
            0,
            1,
            3,
        )

        self.save_dir_backgroundtxtbox = QtWidgets.QLineEdit(self)
        self.save_dir_backgroundtxtbox.setPlaceholderText("Saving directory")
        self.save_dir_backgroundtxtbox.setToolTip(
            "select root folder to scan for .tif files for background selection. After background selection is completed a cell_locations.csv file will be created with the location of the selected backgrounds"
        )
        self.save_dir_backgroundtxtbox.returnPressed.connect(
            self.background_save_dir
        )

        vbox_background.addWidget(self.save_dir_backgroundtxtbox, 1, 0)

        backgroundToolButtonOpenDialog = QtWidgets.QPushButton("Select folder")
        backgroundToolButtonOpenDialog.clicked.connect(
            self.open_background_dir_dialog
        )
        vbox_background.addWidget(backgroundToolButtonOpenDialog, 1, 3)

        backgroundToolButtonRun = QtWidgets.QPushButton(
            "Run Background Selection"
        )
        backgroundToolButtonRun.clicked.connect(self.run_background_selection)
        vbox_background.addWidget(backgroundToolButtonRun, 2, 0, 1, 4)

        vbox_background.addWidget(QLabel("Progress:"), 3, 0)

        self.pbar_background = QProgressBar(self)
        vbox_background.addWidget(self.pbar_background, 4, 0, 1, 4)
        self.updateProgressBackground(0)

        background_select_box.setLayout(vbox_background)

        batch_box = QGroupBox("Step 2 - Batch Analysis selection")

        vbox_batch = QGridLayout()
        vbox_batch.addWidget(
            QLabel(
                "Using the root folder selected in step 1. All .tiff files will be analysed creating a patch analysis folder in each subdirectory using the selected background location as given in step 1. \nThere is a currently known memmory issue. So once memory is filled the program must be restarted and the root folder has to be reselected in step 1. \nNOTE: FILES MUST BE NAMED FPS.TIF EG FOR A 500 FPS CAMERA RECORDING THE FILE IN THE FOLDER SHOULD BE NAMED 500.TIF \nFILES WILL BE SKIPPED IF THE FOLDER ALREADY CONTAINS A PATCH ANALYSIS FOLDER \nFOR CUSTOM FPS SELECT VALUE OTHER THEN 0 \nPARENT FOLDER MUST CONTAIN THE VOLTAGE EG 5 HZ SQUARE CURRENTLY ONLY WORKS FOR 1 OR 5V (1HZ / 5HZ)"
            ),
            0,
            0,
            1,
            3,
        )

        self.checkboxDefaultFPSBatch = QCheckBox("Default fps (500/1000)")
        vbox_batch.addWidget(self.checkboxDefaultFPSBatch, 2, 0)

        self.checkboxDefaultFPSBatch.setChecked(True)

        self.checkboxPhotocurrentBatch = QCheckBox(
            "Batch photocurrent analysis"
        )
        vbox_batch.addWidget(self.checkboxPhotocurrentBatch, 2, 1)

        self.checkboxPhotocurrentBatch.setChecked(True)

        self.customFPS = QSpinBox(self)
        self.customFPS.setMaximum(2000)
        self.customFPS.setSingleStep(250)
        vbox_batch.addWidget(self.customFPS, 2, 4)
        vbox_batch.addWidget(QLabel("Custom FPS:"), 2, 3)

        batchToolButtonRun = QtWidgets.QPushButton("Run Batch Analysis")
        batchToolButtonRun.clicked.connect(self.run_batch)
        vbox_batch.addWidget(batchToolButtonRun, 3, 0, 1, 5)

        vbox_batch.addWidget(QLabel("Progress:"), 4, 0)

        self.pbar_batch = QProgressBar(self)
        vbox_batch.addWidget(self.pbar_batch, 5, 0, 1, 5)
        self.updateProgressBatch(0)

        batch_box.setLayout(vbox_batch)

        stat_box = QGroupBox("Step 3 - Statistical Analysis selection")
        vbox_stat = QGridLayout()

        vbox_stat.addWidget(
            QLabel(
                "Using the root folder selected in step 1. Running statistical analysis will result in the program searching for all statistics files and producing a csv in the root folder containing all statistics."
            ),
            0,
            0,
            1,
            3,
        )

        self.checkboxPhotocurrent = QCheckBox(
            "Include photocurrent statistics"
        )
        vbox_stat.addWidget(self.checkboxPhotocurrent, 1, 0)

        self.checkboxPhotocurrent.setChecked(True)

        # vbox_stat.

        statToolButtonRun = QtWidgets.QPushButton("Run Statistical Analysis")
        statToolButtonRun.clicked.connect(self.run_stat)
        vbox_stat.addWidget(statToolButtonRun, 2, 0, 1, 3)

        vbox_stat.addWidget(QLabel("Progress:"), 3, 0)

        self.pbar_stat = QProgressBar(self)
        vbox_stat.addWidget(self.pbar_stat, 4, 0, 1, 3)
        self.updateProgressStat(0)

        stat_box.setLayout(vbox_stat)

        Display_Container_tabs_tab5 = QWidget()
        Display_Container_tabs_tab5_layout = QGridLayout()

        Display_Container_tabs_tab5_layout.addWidget(
            background_select_box, 0, 0
        )
        Display_Container_tabs_tab5_layout.addWidget(batch_box, 1, 0)
        Display_Container_tabs_tab5_layout.addWidget(stat_box, 2, 0)

        Display_Container_tabs_tab5.setLayout(
            Display_Container_tabs_tab5_layout
        )

        # New code Ian
        Display_Container_tabs_tab6 = QWidget()
        Display_Container_tabs_tab6_layout = QGridLayout()

        settings_box = QGroupBox("Tif and .npy cutter")
        vsettings_box = QGridLayout()

        tif_cutter_example = QLabel(
            "First use filepath to select the directory in which you want to cut the tif and npy files. Select the fps and click run.\n"
            "The program will automatically cut all tif files of more than 5000 frames into multiple videos of 5 seconds (note that missing frames at the start as a result of camertrigger issues are included in the first video)\n"
            "The program also cuts .npy files into smaller files corresponding to the cut tif files"
        )

        vsettings_box.addWidget(tif_cutter_example, 1, 0, 1, 3)

        self.video_dirtxtbox = QtWidgets.QLineEdit(self)
        self.video_dirtxtbox.setPlaceholderText("File path to .tif")
        self.video_dirtxtbox.setToolTip("select .tif video to process")

        vsettings_box.addWidget(self.video_dirtxtbox, 2, 0)

        videoDirToolButtonOpenDialog = QtWidgets.QPushButton(
            "Select directory"
        )
        videoDirToolButtonOpenDialog.clicked.connect(
            self.select_video_directory
        )
        vsettings_box.addWidget(videoDirToolButtonOpenDialog, 2, 2)

        self.fps_recording = QSpinBox(self)
        self.fps_recording.setMaximum(2000)
        self.fps_recording.setSingleStep(250)
        vsettings_box.addWidget(self.fps_recording, 3, 1)
        vsettings_box.addWidget(QLabel("FPS of recorded video:"), 3, 0)

        # Cutting regular GR recordings
        videocutterToolButtonRun = QtWidgets.QPushButton(
            "Split 35 sec.tif files"
        )
        videocutterToolButtonRun.clicked.connect(self.run_video_cutter)
        vsettings_box.addWidget(videocutterToolButtonRun, 3, 2)

        videocutterToolButtonSplit = QtWidgets.QPushButton(
            "Split 35 sec .npy files"
        )
        videocutterToolButtonSplit.clicked.connect(self.run_npy_splitter)
        vsettings_box.addWidget(videocutterToolButtonSplit, 3, 3)

        # Cutting GR FRET measurements
        videocutterToolButtonRun_FRET = QtWidgets.QPushButton(
            "Split GR_FRET .tif files"
        )
        videocutterToolButtonRun_FRET.clicked.connect(
            self.run_GR_FRET_video_cutter
        )
        vsettings_box.addWidget(videocutterToolButtonRun_FRET, 4, 2)

        videocutterToolButtonSplit_FRET = QtWidgets.QPushButton(
            "Split GR_FRET .npy files"
        )
        videocutterToolButtonSplit_FRET.clicked.connect(
            self.run_GR_FRET_npy_splitter
        )
        vsettings_box.addWidget(videocutterToolButtonSplit_FRET, 4, 3)

        videocutterToolButtonRun_FRET = QtWidgets.QPushButton(
            "Split 40 sec (incl 1Hz square) .tif files"
        )
        videocutterToolButtonRun_FRET.clicked.connect(
            self.run_40sec_video_cutter
        )
        vsettings_box.addWidget(videocutterToolButtonRun_FRET, 5, 2)

        videocutterToolButtonSplit_FRET = QtWidgets.QPushButton(
            "Split 40 sec (incl 1Hz square) .npy files"
        )
        videocutterToolButtonSplit_FRET.clicked.connect(
            self.run_40sec_npy_splitter
        )
        vsettings_box.addWidget(videocutterToolButtonSplit_FRET, 5, 3)

        videocutterToolButtonRun_FRET = QtWidgets.QPushButton(
            "Split 5 sec spectral 5Hz square .tif files"
        )
        videocutterToolButtonRun_FRET.clicked.connect(
            self.run_5sec_spectral5hz_video_cutter
        )
        vsettings_box.addWidget(videocutterToolButtonRun_FRET, 6, 2)

        videocutterToolButtonSplit_FRET = QtWidgets.QPushButton(
            "Split 5 sec spectral 5Hz square .npy files"
        )
        videocutterToolButtonSplit_FRET.clicked.connect(
            self.run_5sec_spectral5hz_npy_splitter
        )
        vsettings_box.addWidget(videocutterToolButtonSplit_FRET, 6, 3)

        self.pbar_TIFCutter = QProgressBar(self)
        vsettings_box.addWidget(self.pbar_TIFCutter, 7, 0, 1, 3)
        self.updateProgressTIFCutter(0)

        settings_box.setLayout(vsettings_box)
        Display_Container_tabs_tab6_layout.addWidget(settings_box, 0, 0)

        NIdaq_box = QGroupBox("Multi-NIdaq analysis")
        vNIdaq_box = QGridLayout()

        vNIdaq_box.addWidget(
            QLabel(
                "First select the path to the .npy NIdaq trace. Then select the sampling rate. Select the cut-off before the first start of baseline and signal and select their length. \nAnd also select when the last signal ends (leaving it at 0 means that it will take it all the way to the end)"
            ),
            0,
            0,
            1,
            9,
        )

        self.NIdaq_dirtxtbox = QtWidgets.QLineEdit(self)
        self.NIdaq_dirtxtbox.setPlaceholderText(
            "File path to .npy NIdaq trace"
        )
        self.NIdaq_dirtxtbox.setToolTip("select NIDaq trace to process")

        vNIdaq_box.addWidget(self.NIdaq_dirtxtbox, 1, 0, 1, 9)

        NIdaqDirToolButtonOpenDialog = QtWidgets.QPushButton("Select trace")
        NIdaqDirToolButtonOpenDialog.clicked.connect(self.NIdaq_dir)
        vNIdaq_box.addWidget(NIdaqDirToolButtonOpenDialog, 1, 9)

        vNIdaq_box.addWidget(QLabel("sampling rate: "), 2, 0)

        self.multi_waveform_samplingrate_box = QSpinBox(self)
        self.multi_waveform_samplingrate_box.setMaximum(250000)
        self.multi_waveform_samplingrate_box.setValue(50000)
        self.multi_waveform_samplingrate_box.setSingleStep(500)

        vNIdaq_box.addWidget(self.multi_waveform_samplingrate_box, 2, 1)

        vNIdaq_box.addWidget(QLabel("cut-off (start) [ms]: "), 2, 2)

        self.cut_off_start = QSpinBox(self)
        self.cut_off_start.setValue(0)
        self.cut_off_start.setMaximum(250000)
        self.cut_off_start.setSingleStep(1000)

        vNIdaq_box.addWidget(self.cut_off_start, 2, 3)

        vNIdaq_box.addWidget(QLabel("cut-off (end) [ms]: "), 2, 4)

        self.cut_off_end = QSpinBox(self)
        self.cut_off_end.setValue(0)
        self.cut_off_end.setMaximum(250000)
        self.cut_off_end.setSingleStep(1000)

        vNIdaq_box.addWidget(self.cut_off_end, 2, 5)

        vNIdaq_box.addWidget(QLabel("baseline length [ms]: "), 2, 6)

        self.baseline_length = QSpinBox(self)
        self.baseline_length.setValue(0)
        self.baseline_length.setSingleStep(250)
        self.baseline_length.setMaximum(250000)

        vNIdaq_box.addWidget(self.baseline_length, 2, 7)

        vNIdaq_box.addWidget(QLabel("signal length [ms]: "), 2, 8)

        self.signal_length = QSpinBox(self)
        self.signal_length.setValue(0)
        self.signal_length.setSingleStep(250)
        self.signal_length.setMaximum(250000)

        vNIdaq_box.addWidget(self.signal_length, 2, 9)

        multiNIdaqToolButtonRun = QtWidgets.QPushButton("Run")
        multiNIdaqToolButtonRun.clicked.connect(self.run_multi_NIdaq)
        vNIdaq_box.addWidget(multiNIdaqToolButtonRun, 3, 0, 1, 9)

        self.pbar_multiNIdaq = QProgressBar(self)
        vNIdaq_box.addWidget(self.pbar_multiNIdaq, 4, 0, 1, 9)
        self.updateProgress_multiNIdaq(0)

        NIdaq_box.setLayout(vNIdaq_box)
        Display_Container_tabs_tab6_layout.addWidget(NIdaq_box, 2, 0)

        # @@
        ramp_box = QGroupBox("Ramp fluorescence analysis")
        vramp_box = QGridLayout()

        ramp_example = QLabel(
            "Select .npy file containing fluorescence as a function of time. Set frequency of ramp, fps and ramp max and min values "
        )

        vramp_box.addWidget(ramp_example, 0, 0, 1, 7)

        self.ramp_dirtxtbox = QtWidgets.QLineEdit(self)
        self.ramp_dirtxtbox.setPlaceholderText(
            "File path to .npy ramp fluorescence"
        )
        self.ramp_dirtxtbox.setToolTip("select fluorescence trace to process")

        vramp_box.addWidget(self.ramp_dirtxtbox, 1, 0, 1, 9)

        rampDirToolButtonOpenDialog = QtWidgets.QPushButton(
            "Select fluorescence trace"
        )
        rampDirToolButtonOpenDialog.clicked.connect(self.ramp_dir)
        vramp_box.addWidget(rampDirToolButtonOpenDialog, 1, 9)

        vramp_box.addWidget(QLabel("ramp frequency [Hz]: "), 2, 0)

        self.ramp_freq = QSpinBox(self)
        self.ramp_freq.setValue(1)
        self.ramp_freq.setSingleStep(250)
        self.ramp_freq.setMaximum(250000)

        vramp_box.addWidget(self.ramp_freq, 2, 1)

        vramp_box.addWidget(QLabel("fps [/s]: "), 2, 2)

        self.ramp_fps = QSpinBox(self)
        self.ramp_fps.setMaximum(250000)
        self.ramp_fps.setValue(1000)
        self.ramp_fps.setSingleStep(250)

        vramp_box.addWidget(self.ramp_fps, 2, 3)

        vramp_box.addWidget(QLabel("Voltage min, max [mV]: "), 2, 4)

        self.ramp_volt_min = QSpinBox(self)
        self.ramp_volt_min.setSingleStep(250)
        self.ramp_volt_min.setMaximum(250000)
        self.ramp_volt_min.setMinimum(-250000)
        self.ramp_volt_min.setValue(-1000)

        vramp_box.addWidget(self.ramp_volt_min, 2, 5)

        self.ramp_volt_max = QSpinBox(self)
        self.ramp_volt_max.setSingleStep(250)
        self.ramp_volt_max.setMaximum(250000)
        self.ramp_volt_max.setValue(1000)

        vramp_box.addWidget(self.ramp_volt_max, 2, 6)

        vramp_box.addWidget(QLabel("Avg. filtering kernel size: "), 2, 7)
        self.ramp_avg_filt_kernel_size = QSpinBox(self)
        self.ramp_avg_filt_kernel_size.setSingleStep(1)
        self.ramp_avg_filt_kernel_size.setMaximum(250000)
        self.ramp_avg_filt_kernel_size.setValue(20)

        vramp_box.addWidget(self.ramp_avg_filt_kernel_size, 2, 8)

        rampToolButtonRun = QtWidgets.QPushButton("Run")
        rampToolButtonRun.clicked.connect(self.run_ramp_analysis)
        vramp_box.addWidget(rampToolButtonRun, 2, 9)

        ramp_box.setLayout(vramp_box)
        Display_Container_tabs_tab6_layout.addWidget(ramp_box, 3, 0)

        # Display_Container_tabs_tab6_layout.addWidget()
        Display_Container_tabs_tab6.setLayout(
            Display_Container_tabs_tab6_layout
        )

        # - End of Ian's automization tab -

        # Add tabs
        Display_Container_tabs.addTab(
            Display_Container_tabs_Galvo_WidgetInstance, "Patch clamp display"
        )
        # Display_Container_tabs.addTab(Display_Container_tabs_tab2,"Patch display")
        Display_Container_tabs.addTab(
            Display_Container_tabs_tab3, "Patch perfusion"
        )
        # Display_Container_tabs.addTab(self.Display_Container_tabs_Cellselection,"Cell selection")
        Display_Container_tabs.addTab(
            Display_Container_tabs_tab4, "Display NIdaq trace"
        )
        # @@
        Display_Container_tabs.addTab(
            Display_Container_tabs_tab5, "Batch Analysis"
        )

        Display_Container_tabs.addTab(
            Display_Container_tabs_tab6, "Multi-experiment tools"
        )

        Display_Layout.addWidget(Display_Container_tabs, 0, 0)
        Display_Container.setLayout(Display_Layout)

        self.layout.addWidget(readimageContainer, 0, 0, 1, 2)
        self.layout.addWidget(Display_Container, 1, 0, 1, 2)

    # master_data_analysis.addWidget(imageanalysis_average_Container, 2, 0, 1,1)
    # master_data_analysis.addWidget(imageanalysis_weight_Container, 2, 1, 1,1)

    # === Functions for Data analysis Tab ===

    # - Start of Ian's functions -

    # - Progress bars -
    def updateProgressBackground(self, current_val):
        self.pbar_background.setValue(current_val)

    def updateProgressBatch(self, current_val):
        self.pbar_batch.setValue(current_val)

    def updateProgressStat(self, current_val):
        self.pbar_stat.setValue(current_val)

    def updateProgressTIFCutter(self, current_val):
        self.pbar_TIFCutter.setValue(current_val)

    def updateProgress_multiNIdaq(self, current_val):
        self.pbar_multiNIdaq.setValue(current_val)

    # - Set directory -
    def background_save_dir(self):
        """
        Function written by Ian

        Used for setting the save directory of background selection. Same functions will be used for other steps.
        """
        self.background_save_directory = str(
            self.save_dir_backgroundtxtbox.text()
        )

    def open_background_dir_dialog(self):
        self.background_dir = str(QtWidgets.QFileDialog.getExistingDirectory())
        self.save_dir_backgroundtxtbox.setText(self.background_dir)

    def video_dir(self):
        self.video_dir_variable, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select .tif", "", "All Files (*);; TIF (*.tif)"
        )
        if self.video_dir_variable:
            self.video_dirtxtbox.setText(self.video_dir_variable)

    def select_video_directory(self):
        """
        Opens a dialog to select a directory (for batch .tif processing).
        """
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Directory"
        )

        if directory:
            self.video_dirtxtbox.setText(directory)
            logging.info(f"Selected directory: {directory}")

    def NIdaq_dir(self):
        self.NIdaq_dir_variable, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select NIdaq trace (.npy)",
            "",
            "All Files (*);; numpy (*.npy)",
        )
        if self.NIdaq_dir_variable:
            self.NIdaq_dirtxtbox.setText(self.NIdaq_dir_variable)

    def ramp_dir(self):
        self.ramp_dir_variable, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select ramp fluorecence trace (.npy)",
            "",
            "All Files (*);; numpy (*.npy)",
        )
        if self.ramp_dir_variable:
            self.ramp_dirtxtbox.setText(self.ramp_dir_variable)

    # - Utility functions -
    def search_for_tif_files(self, directory: str) -> list:
        """
        This function searches a directory for every .tif file and returns the full directory path
        :param directory: str: the full directory path to search for .tif files
        :return tif_files: list: list of full directory paths to .tif files in the main directory:
        :return tif_files_path: list: list of .tif files in the main directory
        """
        tif_files: list = []
        tif_files_path: list = []
        folder_path: list = []

        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith("fps.TIF") or file.endswith("fps.tif"):
                    folder_path.append(root)
                    tif_files_path.append(os.path.join(root, file))
                    # x = "data\\"
                    # name = f'{root.split(x)[1]} {file[:-4]}'
                    name = (root.replace(directory, "") + " " + file[:-4])[1:]
                    # print("root, file:", root, "\n", file, "\n", directory)
                    # print("name:", name)

                    tif_files.append(name)

        tif_files = [x.replace("\\", " ") for x in tif_files]
        return tif_files, tif_files_path, folder_path

    def batch_settings(
        self,
        coords: tuple,
        fps: int,
        photocurrent_int: int,
        volt_freq_step: int,
    ):
        # set background location
        self.background_location = coords
        if photocurrent_int == 0 or photocurrent_int == 2:
            self.switch_Vp_or_camtrace.setCurrentIndex(photocurrent_int)
            self.roi_average = pg.PolyLineROI(
                [[0, 0], [0, 30], [30, 30], [30, 0]], closed=True
            )
            self.roi_average = pg.RectROI(
                [coords[0], coords[1]],
                [30, 30],
                centered=True,
                sideScalers=True,
            )
            self.pw_averageimage.view.addItem(self.roi_average)
            self.Spincamsamplingrate.setValue(int(fps))

            self.VoltageStepFrequencyQSpinBox.setValue(int(volt_freq_step))

    def run_batch_sequence(
        self,
        coords: tuple,
        fps: int,
        photocurrent: bool,
        volt_freq_step: int,
        file_path: str,
    ):
        # run required path
        self.batch_settings(coords, fps, photocurrent, volt_freq_step)

        logging.info("file loaded")

        logging.info("========== finished loading settings ==========")
        self.getfile_automated(file_path)
        logging.info("========== finished fetching file ==========")
        logging.info("========== starting analysis ==========")
        self.finish_analysis()

    # - Start complete functions -
    def run_background_selection(self):
        """
        Function retrofitted from matig_of_handmatig module
        """
        logging.info("Running background selection")
        directory = self.background_dir
        matig_of_handmatig.main(directory)
        self.updateProgressBackground(100)

    # @@@ TODO
    def run_batch(self):
        directory = self.background_dir

        def main():
            """
            :param: directory : str : set directory containing all the data (.tif files).

            Limitations:
            (1): program takes the root location of the data: in case of data stored in the shape of data\\cell x\\observation x take the directory refering to ...\\data\\
            (2): program will make a patch analysis file for the tif it finds. So make sure that there is only one .tif file in each folder.

            """

            files, paths, folder_path = self.search_for_tif_files(directory)

            fps = []
            photocurrent = []

            succesful = []
            failed = []
            already_finished = []

            # idk why this works, but does not work doing it 3x
            with open(f"{directory}/cell_locations.csv", mode="r") as csv_file:
                reader = csv.reader(csv_file)
                col = [row[0] for row in reader]  # [1:]
            with open(f"{directory}/cell_locations.csv", mode="r") as csv_file:
                reader = csv.reader(csv_file)
                x_cord = [row[1] for row in reader]  # [1:]
            with open(f"{directory}/cell_locations.csv", mode="r") as csv_file:
                reader = csv.reader(csv_file)
                y_cord = [row[2] for row in reader]  # [1:]

            logging.info(f"column: {col}")
            logging.info(f"x_cord: {x_cord}")
            logging.info(f"y_cord: {y_cord}")

            if not self.checkboxPhotocurrentBatch.isChecked():
                logging.info("photocurrent analysis has been turned off")

            maxValue = len(files)
            self.updateProgressBatch(0)

            for i, file in enumerate(files):
                # print information
                logging.info(
                    "------------------ itteration information: ------------------"
                )
                logging.info(f"iteration number: {i}; file: {file}")
                # print(f'coordinates (x,y): ({x_cord[i]}, {y_cord[i]})')

                # pbar
                processValue = round(i / maxValue * 100)
                self.updateProgressBatch(processValue)

                # add fps to fps list
                if self.checkboxDefaultFPSBatch.isChecked() and "500" in file:
                    fps.append("500")
                elif (
                    self.checkboxDefaultFPSBatch.isChecked() and "1000" in file
                ):
                    fps.append("1000")
                elif (
                    self.customFPS.value() != 0
                    and str(self.customFPS.value()) in file
                ):
                    fps.append(str(self.customFPS.value()))

                # add photocurrent status
                if (
                    "Photocurrent" or "photocurrent"
                ) in file and self.checkboxPhotocurrentBatch.isChecked():
                    photocurrent.append("2")
                    logging.info(
                        "------------------ PHOTOCURRENT ANALYSIS ------------------"
                    )
                elif (
                    "Photocurrent" or "photocurrent"
                ) in file and not self.checkboxPhotocurrentBatch.isChecked():
                    photocurrent.append("-1")
                    logging.info(
                        "------------------ PHOTOCURRENT ANALYSIS TURNED OFF -> SKIPPED ------------------"
                    )
                if not ("Photocurrent" or "photocurrent") in file:
                    photocurrent.append("0")

                logging.info(f"fps: {fps[i]}")

                # Check if the current directory is "5Hz square" or "Regular Photocurrent"
                current_directory_name = os.path.basename(folder_path[i])

                # Use this if you don't want to analyze everything that starts with "5Hz square"
                # if current_directory_name in ["532nm and 639nm 5Hz square"]:
                # Use this if you don want to analyze everything that starts with "5Hz square"
                if current_directory_name.startswith(
                    "5Hz square"
                ) or current_directory_name in [
                    "5Hz square",
                    "Regular Photocurrent",
                ]:
                    logging.info(
                        f"Processing directory: {current_directory_name}"
                    )
                    x = folder_path[i].replace(
                        folder_path[i].split("\\")[-1], ""
                    )[:-1]
                    fol_path_in_col = x  # f'\'{x}\''
                    # print("path to cell", fol_path_in_col)

                    # check if already analysed
                    if fol_path_in_col in col:
                        patch_analysis_path = (
                            str(folder_path[i]) + "\\Patch analysis\\"
                        )
                        # print(patch_analysis_path)
                        if os.path.exists(str(patch_analysis_path)):
                            file_count = len(os.listdir(patch_analysis_path))
                        else:
                            file_count = 0
                        if photocurrent[i] == "2":
                            photo_cur_dir = os.listdir(folder_path[i])
                            length = len(
                                [
                                    x
                                    for x in photo_cur_dir
                                    if "Photo-current" in x
                                ]
                            )
                            if length > 0:
                                file_count = 27

                        # if not yet analysed continue:
                        if file_count < 26:
                            logging.info("found")

                            logging.info(
                                f"number of already existing patch analysis files: {file_count}"
                            )
                            logging.info(
                                f"Patch analysis folder: {patch_analysis_path}"
                            )
                            # print(col.index(fol_path_in_col))

                            cell_loc = [
                                int(float(x_cord[col.index(fol_path_in_col)])),
                                int(float(y_cord[col.index(fol_path_in_col)])),
                            ]
                            # print(f'{file} : background location : {cell_loc} : photocurrent {photocurrent[i]} (0 False / 2 True)')
                            # save_background(paths[i], folder_path[i], cell_loc)
                            voltage = 0
                            if (
                                "5" in folder_path[i]
                                and "Hz" in folder_path[i]
                            ):
                                voltage = 5
                                logging.info("5 Hz signal")
                            elif (
                                "1" in folder_path[i]
                                and "Hz" in folder_path[i]
                            ):
                                voltage = 1
                                logging.info("1 Hz signal")
                            logging.info(
                                f"folder path: {folder_path[i]}, Hz {voltage}"
                            )

                            # TODO: implement 5Hz / 1Hz taken from file name => voltage frequency step = 1 / 5Hz, photocurrent lezen en begrijpen
                            # @@ TODO FIX THAT ALSO WORKS FOR PHOTOCURRENT

                            logging.info(
                                "photocurrent analysis status:  ",
                                photocurrent[i],
                            )
                            photo_as_int = int(photocurrent[i])
                            if photo_as_int != -1:
                                self.run_batch_sequence(
                                    cell_loc,
                                    fps[i],
                                    photo_as_int,
                                    voltage,
                                    paths[i].replace("\\", "/"),
                                )
                                time.sleep(0.1)
                                succesful.append(folder_path[i])
                            else:
                                logging.info(fol_path_in_col)
                                already_finished.append(folder_path[i])
                        else:
                            logging.info(fol_path_in_col)
                            already_finished.append(folder_path[i])

                    else:
                        # continue
                        logging.error(col[1])
                        logging.error(fol_path_in_col)
                        logging.error(
                            "---------- ERROR: not in csv file ----------"
                        )

                else:  # except:
                    logging.error(folder_path[i])
                    logging.error("-------- general error --------")
                    # print(e)
                    failed.append(folder_path[i])

            logging.info("skipped: ", already_finished)
            logging.info("succesful", succesful)
            logging.info("failed", failed)

            self.updateProgressBatch(100)

        main()

    def run_stat(self):
        def clean_string(s):
            # Use regular expression to keep only numbers and dot
            cleaned_string = re.sub(r"[^0-9.]", "", s)
            return cleaned_string

        def search_for_txt_files(directory: str) -> list:
            """
            This function searches a directory for every .txt file and returns the full directory path
            :param directory: str: the full directory path to search for .txt files
            :return txt_files: list: list of full directory paths to .txt files in the main directory:
            :return txt_files_path: list: list of .txt files in the main directory
            """
            txt_files: list = []
            txt_files_path: list = []
            folder_path: list = []

            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(".txt"):
                        folder_path.append(root)
                        txt_files_path.append(os.path.join(root, file))
                        # x = "data\\"
                        # name = f'{root.split(x)[1]} {file[:-4]}'
                        name = (root.replace(directory, "") + " " + file[:-4])[
                            1:
                        ]
                        txt_files.append(name)

            txt_files = [x.replace("\\", " ") for x in txt_files]
            return txt_files, txt_files_path, folder_path

        def search_for_x_files(directory: str, ends_with: str) -> list:
            """
            This function searches a directory for every photocurrent file and returns the full directory path
            :param directory: str: the full directory path to search for photucrrent files
            :return photucrrent_files: list: list of full directory paths to photocurrent files in the main directory:
            :return photucrrent_files_path: list: list of photocurrent files in the main directory
            """
            files: list = []
            files_path: list = []
            folder_path: list = []

            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(f"{ends_with}"):
                        folder_path.append(root)
                        files_path.append(os.path.join(root, file))
                        # x = "data\\"
                        # name = f'{root.split(x)[1]} {file[:-4]}'
                        name = (root.replace(directory, "") + " " + file[:-4])[
                            1:
                        ]
                        files.append(name)

            files = [x.replace("\\", " ") for x in files]
            return files, files_path, folder_path

        def txt_to_list(file_path: str) -> list:
            """
            This function reads a .txt file and converts it to a list where each element is a line in the .txt file
            :param file_path: str: the full directory path to the .txt file
            :return: list: list where each element is a line in the .txt file
            """
            with open(file_path, "r") as file:
                lines = file.readlines()
            # logger.debug(f'file: {file_path} converted to list')
            return lines

        def extract_data_from_string(string: str) -> None:
            """
            Takes the string, splits it by '=' and prints it
            :param string:
            :return:
            """
            if "=" not in string:
                return None
            col_name, col_val = string.split("=")[0], string.split("=")[1]
            # remove leading and trailing whitespaces
            col_name = col_name.strip()
            col_val = col_val.strip()
            if "," in col_val:
                col_val_1, col_val_2 = col_val.split(",")
                col_name_1 = f"{col_name} [ms]"
                col_name_2 = f"{col_name} [%]"
                col_val_1 = col_val_1.strip()
                col_val_2 = col_val_2.strip()
                col_name_1 = col_name_1.strip()
                col_name_2 = col_name_2.strip()
                return col_name_1, col_name_2, col_val_1, col_val_2
            else:
                return col_name, col_val

        def extract_data_from_list(list_of_strings: list) -> list:
            """
            This function gets a list of strings. For each string it will look for '=' if it does not contain '=' it will be skipped.
            If it does contain '=' then it will take the string which will be in shape of column_name = column_value_1 unit_1, column_value_2 unit_2, ... column_value_n unit_n
            For each string it will create the pairs of column_name unit_1 and column_name unit_2 and so on. It will return a list of column names and a list of column values
            :param list_of_strings: list: list of strings where each element is a line in the .txt file
            :return: list: list of strings where each element is a line in the .txt file containing '='
            """
            column_name = []
            column_value = []
            for string in list_of_strings:
                if "=" not in string:
                    continue
                if "," in string:
                    col_name_1, col_name_2, col_val_1, col_val_2 = (
                        extract_data_from_string(string)
                    )
                    column_name.append(col_name_1)
                    column_name.append(col_name_2)
                    column_value.append(col_val_1)
                    column_value.append(col_val_2)
                else:
                    col_name, col_val = extract_data_from_string(string)
                    column_name.append(col_name)
                    column_value.append(col_val)

            return column_name, column_value

        # Function to convert list to pandas dataframe. Takes a list of strings where each element is the beginning of the line in the .txt file. Creates a dataframe with the first element as the column name and the rest of the elements as the rows
        def list_to_dataframe(
            name: str, path_cell: str, list_of_strings: list, index: int = 0
        ) -> pd.DataFrame:
            """
            This function takes a list of strings and converts it to a pandas dataframe. The first element is the column name, then the path to the txt file and the rest of the elements are the rows
            :param name: str: name of the cell
            :param list_of_strings: list: list of strings where each element is the beginning of the line in the .txt file
            :param path_cell: str: the full directory path to the .txt file
            :return: pd.DataFrame: dataframe with the first element as the column name and the rest of the elements as the rows. Note that the first element is the name of the cell
            """

            column_name, column_value = extract_data_from_list(list_of_strings)

            pair = []
            if index == 0:
                for i in range(len(column_value)):
                    pair.append((column_name[i], column_value[i]))
            else:
                for i in range(len(column_value)):
                    pair.append((column_value[i]))

            # dataframe first column is cell name, second column is path to the cell file and then followed with column name and column value
            df = pd.DataFrame(pair)

            df = df.T
            df.insert(0, "Cell", name.replace(" ", "_"))
            # df.insert(1, 'Path', path_cell.replace(' ', '_'))

            index = 1
            # print("df", df)
            return df

        # Finished
        def merge_dataframes(list_of_dataframes: list) -> pd.DataFrame:
            """
            This function merges a list of dataframes together
            :param list_of_dataframes: list: list of dataframes to merge
            :return: pd.DataFrame: merged dataframe
            """
            # Takes the first dataframe from a list of dataframes and keeps adding the last row of other dataframes in the list
            return pd.concat(list_of_dataframes)  # , axis=1)

        # function to convert .txt files to dataframe. With optional save as .csv file
        def txt_to_dataframe(
            directory: str, save_as_csv: bool = False
        ) -> pd.DataFrame:
            """
            This function converts .txt files to a pandas dataframe
            :param directory: str: the full directory path to search for .txt files
            :param save_as_csv: bool: whether to save the dataframe as a .csv file
            :return: pd.DataFrame: dataframe of all .txt files
            """

            txt_files, txt_files_path, folder_path = search_for_txt_files(
                directory
            )
            # logger.info(f'found {len(txt_files)} .txt files in {directory}')
            logging.info("\n")
            cell_name = [x.split(" Patch analysis")[0] for x in txt_files]

            list_of_dataframes = []

            max_val = len(txt_files_path)
            self.updateProgressStat(0)
            time.sleep(0.1)

            for i in range(len(txt_files_path)):
                new_percentage = round(i / max_val * 100)
                self.updateProgressStat(new_percentage)
                # logger.info(f'processing {txt_files_path[i]}')
                lines = txt_to_list(txt_files_path[i])
                df = list_to_dataframe(
                    cell_name[i], txt_files_path, lines, index=i
                )
                list_of_dataframes.append(df)
            df = merge_dataframes(list_of_dataframes)
            df.iat[0, 0] = "Cell"

            if save_as_csv:
                save_dataframe_as_csv(df, "dataframe.csv")
            self.updateProgressStat(100)
            return df

        # Finished
        def save_dataframe_as_csv(
            dataframe: pd.DataFrame, file_name: str
        ) -> None:
            """
            This function saves a dataframe as a .csv file
            :param dataframe: pd.DataFrame: dataframe to save as .csv file
            :param file_name: str: name of the .csv file
            """
            dataframe.to_csv(file_name, index=False)

        # Finished
        def search_for_photocurrent_files(directory: str) -> list:
            """
            This function searches a directory for every photocurrent file and returns the full directory path
            :param directory: str: the full directory path to search for photucrrent files
            :return photucrrent_files: list: list of full directory paths to photocurrent files in the main directory:
            :return photucrrent_files_path: list: list of photocurrent files in the main directory
            """
            photocurrent_files: list = []
            photocurrent_files_path: list = []
            folder_path: list = []

            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith("pA.png"):
                        folder_path.append(root)
                        photocurrent_files_path.append(
                            os.path.join(root, file)
                        )
                        name = (root.replace(directory, "") + " " + file[:-4])[
                            1:
                        ]
                        photocurrent_files.append(name)

            photocurrent_files = [
                x.replace("\\", " ") for x in photocurrent_files
            ]
            return photocurrent_files, photocurrent_files_path, folder_path

        def add_photocurrent_to_csv(directory: str, file_name: str) -> None:
            """
            This function adds photocurrent to the csv file
            :param directory: str: the full directory path to search for photocurrent files
            :param file_name: str: name of the .csv file
            """
            # loads the csv file
            df = pd.read_csv(file_name)
            os.remove(file_name)
            df["Photocurrent [pA]"] = None

            # takes the name of columns
            top_left = df.columns[0]

            photocurrent_files, photocurrent_files_path, folder_path = (
                search_for_photocurrent_files(directory)
            )

            values = [x.split(" ")[-2] for x in photocurrent_files]
            folder = [
                os.path.dirname(x) for x in folder_path
            ]  # ERRORFIX used to be [os.path.dirname(x)[:-1] for x in folder_path]

            folder = [x.replace(f"{directory}\\", "") for x in folder]

            for i in range(len(df)):

                temp = df[f"{top_left}"][i]
                temp_folder = [
                    x.replace("\\", "_").replace(" ", "_") for x in folder
                ]
                for file in temp_folder:
                    if file in temp:
                        index = temp_folder.index(file)
                        df["Photocurrent [pA]"][i] = values[index] + " pA"

            # Ugly solution, but it works. So actually it's beautiful
            new_header = df.iloc[0]
            df.at[0, "Cell"] = "Cell"
            df.at[0, "Photocurrent [pA]"] = "Photocurrent [pA]"

            df = df[1:].reset_index(drop=True)
            df.columns = new_header
            # End of ugly

            logging.info(df)
            file_name = "statistical_analysis_with_photocurrent.csv"
            df.to_csv(f"{directory}/{file_name}", index=False)
            logging.info(f"saved to: {directory}/{file_name}")

        def main_statistics(directory):

            # directory = r'M:/tnw/ist/do/projects/Neurophotonics/Brinkslab/People/Xin Meng/Code/Python_test_TF2/ImageAnalysis/data'

            # logger.info(f'running txt_to_dataframe for directory: {directory}')
            df: pd.DataFrame = txt_to_dataframe(directory, save_as_csv=True)

            logging.info("\n")
            # logger.debug(f'dataframe: {df}')
            # logger.info(f'dataframe shape: {df.shape}')
            # logger.info(f'finished running txt_to_dataframe for directory: {directory}')

            # logger.info(f'running add_photocurrent for dataframe')
            if self.checkboxPhotocurrent.isChecked():
                add_photocurrent_to_csv(directory, "dataframe.csv")
            else:
                file_name = "statistical_analysis.csv"
                os.replace("dataframe.csv", f"{directory}/{file_name}")

                # Another ugly solution, but it works. So actually it's beautiful
                df = pd.read_csv(f"{directory}/{file_name}")
                new_header = df.iloc[0]
                df = df.reset_index(drop=True)
                df.columns = new_header

                logging.info(df)
                df.to_csv(
                    f"{directory}/{file_name}", index=False, header=False
                )
                logging.info(f"saved to: {directory}/{file_name}")
                # End of ugly

            # logger.info(f'finished running add_photocurrent for dataframe')

        # time.sleep(0.1)
        directory = self.background_dir
        main_statistics(directory)

    # def run_video_cutter(self) -> None:
    #     """
    #     This function cuts a video taken at a certain fps into videos of 5 seconds to be analysed.

    #     Note: as a result of known problems with camera trigger, the first few frames might be missed of the recording.
    #     Therefore the videos produced will consider of length x, 5s, 5s etc where x = original_videolength mod 5

    #     :param video_time: int/float: desired time for resulting video. Note video_time * fps must be integer.
    #     """

    def run_video_cutter(self) -> None:
        """
        Recursively searches through folders starting from the selected directory,
        and cuts each multi-frame .tif file into 7 fragments.
        Each fragment is saved to one of 7 hardcoded subdirectories in the same folder. (These subdirectories are made while doing this)
        """

        self.updateProgressTIFCutter(0)

        # Hardcoded 7 directory names
        dir_names = [
            "5Hz square",
            "Regular Photocurrent",
            "Incremental 5Hz square",
            "1Hz ramp",
            "Spectral photocurrent",
            "Spectral 5Hz square",
            "Incremental photocurrent",
        ]

        root_dir: str = str(self.video_dirtxtbox.text())

        if not os.path.isdir(root_dir):
            logging.info("Selected path is not a directory.")
            return

        tif_files = []
        for folder, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith((".tif", ".btf")):
                    full_path = os.path.join(folder, file)
                    try:
                        with TiffFile(full_path) as tif:
                            if len(tif.pages) >= 5001:
                                tif_files.append(full_path)
                    except Exception as exc:
                        logging.info(
                            f"Skipping unreadable file {full_path}",
                            exc_info=exc,
                        )

        total_files = len(tif_files)

        if total_files == 0:
            logging.info("No .tif files found.")
            return

        for file_index, path in enumerate(tif_files):
            try:
                with TiffFile(path) as tif:
                    num_frames = len(tif.pages)

                    if num_frames < 2:
                        logging.info(
                            f"Skipping file with only {num_frames} frame(s): {path}"
                        )
                        continue

                    frames_per_cut = math.ceil(num_frames / 7)
                    base_dir = os.path.dirname(path)

                    for i in range(7):
                        end = num_frames - (i * frames_per_cut)
                        start = max(0, end - frames_per_cut)
                        frames = tif.asarray(key=range(start, end))

                        cut_dir = os.path.join(
                            base_dir, dir_names[6 - i]
                        )  # reverse order
                        os.makedirs(cut_dir, exist_ok=True)

                        out_path = os.path.join(cut_dir, "500 fps.TIF")
                        io.imsave(out_path, frames)
                    progress = round((file_index + 1) / total_files * 100)
                    self.updateProgressTIFCutter(progress)
                    logging.info(f"Finished cutting: {path}")

            except Exception as exc:
                logging.error(f"Error processing {path}", exc_info=exc)

            progress = round((file_index + 1) / total_files * 100)
            self.updateProgressTIFCutter(progress)

        logging.info("All videos successfully cut.")

        """
        # Depreciated version: does not depend on tiffile. But needs lots of ram as it takes the whole tif in ram.
        video_time = 5
        self.updateProgressTIFCutter(0)

        fps: int = self.fps_recording.value()
        path: str = str(self.video_dirtxtbox.text())

        video_length: int = round(video_time * fps)


        original_video: np.ndarray = io.imread(path)

        max_number_of_videos = int(np.shape(original_video)[0]/video_length)
        remainder = int(np.shape(original_video)[0]%video_length)


        # Take videos from the back
        flipped_video = np.flip(original_video, axis=0)
        for i in range(max_number_of_videos+1):
            time.sleep(1)
            if i != max_number_of_videos:
                # take videos of size 5s
                video_size = video_length
            else:
                # last video is remaining video
                video_size = remainder

            new_video = flipped_video[i * video_length:  i * video_length + video_size,:,:]
            new_video = np.flip(new_video, axis=0)

            io.imsave(f'{path[:len(path) - 4]}{max_number_of_videos - i}.TIF', new_video)
            new_percentage = round(i / (max_number_of_videos) * 100)
            self.updateProgressTIFCutter(new_percentage)


    """

    def run_npy_splitter(self) -> None:
        """
        Processes .npy files in the selected directory:
        - Finds one current file (Ip*.npy) and one waveform file (*__Wavefroms_sr_50000.npy)
        - Splits them into 7 segments using hardcoded ranges
        - Saves the segments into 7 subdirectories with fixed names
        """

        self.updateProgressTIFCutter(0)

        """
        current_ranges = [
            (5, 250004),
            (250005, 500004),
            (500005, 750004),
            (750005, 1000004),
            (1000007, 1250006),
            (1250007, 1500006),
            (1500007, 1750004)
        ]

        waveform_ranges = [
            (0, 249999),
            (250000, 499999),
            (500000, 749999),
            (750000, 999999),
            (1000002, 1250001),
            (1250002, 1500001),
            (1500002, 1749999)
        ]
        """

        current_ranges = [
            (5, 250004),
            (250005, 500004),
            (500005, 750004),
            (750005, 1000004),
            (1000005, 1250004),
            (1250005, 1500004),
            (1500005, 1750004),
        ]

        waveform_ranges = [
            (0, 249999),
            (250000, 499999),
            (500000, 749999),
            (750000, 999999),
            (1000000, 1249999),
            (1250000, 1499999),
            (1500000, 1749999),
        ]

        dir_names = [
            "5Hz square",
            "Regular Photocurrent",
            "Incremental 5Hz square",
            "1Hz ramp",
            "Spectral photocurrent",
            "Spectral 5Hz square",
            "Incremental photocurrent",
        ]

        root_dir = str(self.video_dirtxtbox.text())

        if not os.path.isdir(root_dir):
            logging.info("Selected path is not a directory.")
            return

        for folder, _, files in os.walk(root_dir):
            if any(folder.endswith(name) for name in dir_names):
                continue
            for file in files:
                file_path = os.path.join(folder, file)

                if file.endswith(".npy"):
                    if "__Wavefroms_sr_50000" in file:
                        try:
                            waveform_data = np.load(
                                file_path, allow_pickle=True
                            )
                            assert (
                                waveform_data.shape == (8,)
                                and waveform_data.dtype == object
                            )

                            for i, (start, end) in enumerate(waveform_ranges):
                                length = end - start + 1

                                # Build dtype for this slice
                                padded_length = length + 1 + 1
                                dtype = [
                                    ("Waveform", "<f8", (padded_length,)),
                                    ("Sepcification", "<U20"),
                                ]

                                waveform_list = []

                                for j, item in enumerate(waveform_data):
                                    wave = item["Waveform"][
                                        start : end + 1
                                    ]  # Access the waveform (first element of the tuple)
                                    padded_wave = np.concatenate(
                                        [
                                            np.full(1, wave[0]),
                                            wave,
                                            np.full(1, wave[-1]),
                                        ]
                                    )

                                    spec = item["Sepcification"]
                                    dtype = [
                                        (
                                            "Waveform",
                                            padded_wave.dtype,
                                            padded_wave.shape,
                                        ),
                                        ("Sepcification", "<U20"),
                                    ]
                                    record = np.array(
                                        (padded_wave, spec), dtype=dtype
                                    )[
                                        ()
                                    ]  # [()] unpacks to np.void
                                    waveform_list.append(record)

                                out_dir = os.path.join(folder, dir_names[i])
                                os.makedirs(out_dir, exist_ok=True)

                                out_path = os.path.join(out_dir, file)
                                waveform_array = np.array(
                                    waveform_list, dtype=object
                                )
                                np.save(out_path, waveform_array)
                                logging.info(
                                    f"Saved waveform part to {out_path}"
                                )

                        except Exception as exc:
                            logging.info(
                                f"Failed to process waveform file {file_path}",
                                exc_info=exc,
                            )

                    elif file.startswith("Ip") and file.endswith(".npy"):
                        try:
                            current_data = np.load(
                                file_path, allow_pickle=True
                            )
                            assert (
                                current_data.ndim == 1
                                and current_data.dtype == np.float64
                            )

                            for i, (start, end) in enumerate(current_ranges):
                                custom_padding_start = np.array(
                                    [
                                        5.00000000e04,
                                        -1.51955698e-03,
                                        1.61014993e-04,
                                        -5.23595314e-15,
                                        3.38772592e-18,
                                        0,
                                    ],
                                    dtype=np.float64,
                                )

                                padded_current = np.concatenate(
                                    [
                                        custom_padding_start,
                                        current_data[start : end + 1],
                                        np.zeros(1, dtype=np.float64),
                                    ]
                                )

                                out_dir = os.path.join(folder, dir_names[i])
                                os.makedirs(out_dir, exist_ok=True)

                                out_path = os.path.join(out_dir, file)
                                np.save(out_path, padded_current)
                                logging.info(
                                    f"Saved current curve part to {out_path}"
                                )

                        except Exception as exc:
                            logging.info(
                                f"Failed to process current curve file {file_path}",
                                exc_info=exc,
                            )
        logging.info("Finished cutting .npy files")

    def run_GR_FRET_video_cutter(self) -> None:
        """
        run_GR_FRET_video_cutter(self) and run_GR_FRET_npy_splitter(self) are copies of run_video_cutter(self) and
        run_npy_splitter(self), but made to cut tif files with a different structure. The 2 functions here are meant for
        measurements where voltage to the AOTF is increased incrementally from 0.5V to 5V incrementally in 7 steps of 5 seconds.
        """
        """
        Recursively searches through folders starting from the selected directory,
        and cuts each multi-frame .tif file into 7 fragments.
        Each fragment is saved to one of 7 hardcoded subdirectories in the same folder. (These subdirectories are made while doing this)
        """

        self.updateProgressTIFCutter(0)

        # Hardcoded 7 directory names
        dir_names = [
            "5Hz square 0p5V",
            "5Hz square 1p0V",
            "5Hz square 1p5V",
            "5Hz square 2p0V",
            "5Hz square 3p0V",
            "5Hz square 4p0V",
            "5Hz square 5p0V",
        ]

        root_dir: str = str(self.video_dirtxtbox.text())

        if not os.path.isdir(root_dir):
            logging.info("Selected path is not a directory.")
            return

        tif_files = []
        for folder, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith((".tif", ".btf")):
                    full_path = os.path.join(folder, file)
                    try:
                        with TiffFile(full_path) as tif:
                            if len(tif.pages) >= 5001:
                                tif_files.append(full_path)
                    except Exception as exc:
                        logging.info(
                            f"Skipping unreadable file {full_path}",
                            exc_info=exc,
                        )

        total_files = len(tif_files)

        if total_files == 0:
            logging.info("No .tif files found.")
            return

        for file_index, path in enumerate(tif_files):
            try:
                with TiffFile(path) as tif:
                    num_frames = len(tif.pages)

                    if num_frames < 2:
                        logging.info(
                            f"Skipping file with only {num_frames} frame(s): {path}"
                        )
                        continue

                    frames_per_cut = math.ceil(num_frames / 7)
                    base_dir = os.path.dirname(path)

                    for i in range(7):
                        end = num_frames - (i * frames_per_cut)
                        start = max(0, end - frames_per_cut)
                        frames = tif.asarray(key=range(start, end))

                        cut_dir = os.path.join(
                            base_dir, dir_names[6 - i]
                        )  # reverse order
                        os.makedirs(cut_dir, exist_ok=True)

                        out_path = os.path.join(cut_dir, "500 fps.TIF")
                        io.imsave(out_path, frames)
                    progress = round((file_index + 1) / total_files * 100)
                    self.updateProgressTIFCutter(progress)
                    logging.info(f"Finished cutting: {path}")

            except Exception as exc:
                logging.error(f"Error processing {path}", exc_info=exc)

            progress = round((file_index + 1) / total_files * 100)
            self.updateProgressTIFCutter(progress)

        logging.info("All videos successfully cut.")

        """
        # Depreciated version: does not depend on tiffile. But needs lots of ram as it takes the whole tif in ram.
        video_time = 5
        self.updateProgressTIFCutter(0)

        fps: int = self.fps_recording.value()
        path: str = str(self.video_dirtxtbox.text())

        video_length: int = round(video_time * fps)


        original_video: np.ndarray = io.imread(path)

        max_number_of_videos = int(np.shape(original_video)[0]/video_length)
        remainder = int(np.shape(original_video)[0]%video_length)


        # Take videos from the back
        flipped_video = np.flip(original_video, axis=0)
        for i in range(max_number_of_videos+1):
            time.sleep(1)
            if i != max_number_of_videos:
                # take videos of size 5s
                video_size = video_length
            else:
                # last video is remaining video
                video_size = remainder

            new_video = flipped_video[i * video_length:  i * video_length + video_size,:,:]
            new_video = np.flip(new_video, axis=0)

            io.imsave(f'{path[:len(path) - 4]}{max_number_of_videos - i}.TIF', new_video)
            new_percentage = round(i / (max_number_of_videos) * 100)
            self.updateProgressTIFCutter(new_percentage)


    """

    def run_GR_FRET_npy_splitter(self) -> None:
        # See comment above run_GR_FRET_video_cutter()
        """
        Processes .npy files in the selected directory:
        - Finds one current file (Ip*.npy) and one waveform file (*__Wavefroms_sr_50000.npy)
        - Splits them into 7 segments using hardcoded ranges
        - Saves the segments into 7 subdirectories with fixed names
        """

        self.updateProgressTIFCutter(0)

        """
        current_ranges = [
            (5, 250004),
            (250005, 500004),
            (500005, 750004),
            (750005, 1000004),
            (1000007, 1250006),
            (1250007, 1500006),
            (1500007, 1750004)
        ]

        waveform_ranges = [
            (0, 249999),
            (250000, 499999),
            (500000, 749999),
            (750000, 999999),
            (1000002, 1250001),
            (1250002, 1500001),
            (1500002, 1749999)
        ]
        """

        current_ranges = [
            (5, 250004),
            (250005, 500004),
            (500005, 750004),
            (750005, 1000004),
            (1000005, 1250004),
            (1250005, 1500004),
            (1500005, 1750004),
        ]

        waveform_ranges = [
            (0, 249999),
            (250000, 499999),
            (500000, 749999),
            (750000, 999999),
            (1000000, 1249999),
            (1250000, 1499999),
            (1500000, 1749999),
        ]

        dir_names = [
            "5Hz square 0p5V",
            "5Hz square 1p0V",
            "5Hz square 1p5V",
            "5Hz square 2p0V",
            "5Hz square 3p0V",
            "5Hz square 4p0V",
            "5Hz square 5p0V",
        ]

        root_dir = str(self.video_dirtxtbox.text())

        if not os.path.isdir(root_dir):
            logging.info("Selected path is not a directory.")
            return

        for folder, _, files in os.walk(root_dir):
            if any(folder.endswith(name) for name in dir_names):
                continue
            for file in files:
                file_path = os.path.join(folder, file)

                if file.endswith(".npy"):
                    if "__Wavefroms_sr_50000" in file:
                        logging.info("File found")
                        try:
                            waveform_data = np.load(
                                file_path, allow_pickle=True
                            )
                            assert (
                                waveform_data.shape == (4,)
                                and waveform_data.dtype == object
                            )

                            for i, (start, end) in enumerate(waveform_ranges):
                                length = end - start + 1

                                # Build dtype for this slice
                                padded_length = length + 1 + 1
                                dtype = [
                                    ("Waveform", "<f8", (padded_length,)),
                                    ("Sepcification", "<U20"),
                                ]

                                waveform_list = []

                                for j, item in enumerate(waveform_data):
                                    wave = item["Waveform"][
                                        start : end + 1
                                    ]  # Access the waveform (first element of the tuple)
                                    padded_wave = np.concatenate(
                                        [
                                            np.full(1, wave[0]),
                                            wave,
                                            np.full(1, wave[-1]),
                                        ]
                                    )

                                    spec = item["Sepcification"]
                                    dtype = [
                                        (
                                            "Waveform",
                                            padded_wave.dtype,
                                            padded_wave.shape,
                                        ),
                                        ("Sepcification", "<U20"),
                                    ]
                                    record = np.array(
                                        (padded_wave, spec), dtype=dtype
                                    )[
                                        ()
                                    ]  # [()] unpacks to np.void
                                    waveform_list.append(record)

                                out_dir = os.path.join(folder, dir_names[i])
                                os.makedirs(out_dir, exist_ok=True)

                                out_path = os.path.join(out_dir, file)
                                waveform_array = np.array(
                                    waveform_list, dtype=object
                                )
                                np.save(out_path, waveform_array)
                                logging.info(
                                    f"Saved waveform part to {out_path}"
                                )

                        except Exception as exc:
                            logging.info(
                                f"Failed to process waveform file {file_path}",
                                exc_info=exc,
                            )

                    elif file.startswith("Ip") and file.endswith(".npy"):
                        try:
                            current_data = np.load(
                                file_path, allow_pickle=True
                            )
                            assert (
                                current_data.ndim == 1
                                and current_data.dtype == np.float64
                            )

                            for i, (start, end) in enumerate(current_ranges):
                                custom_padding_start = np.array(
                                    [
                                        5.00000000e04,
                                        -1.51955698e-03,
                                        1.61014993e-04,
                                        -5.23595314e-15,
                                        3.38772592e-18,
                                        0,
                                    ],
                                    dtype=np.float64,
                                )

                                padded_current = np.concatenate(
                                    [
                                        custom_padding_start,
                                        current_data[start : end + 1],
                                        np.zeros(1, dtype=np.float64),
                                    ]
                                )

                                out_dir = os.path.join(folder, dir_names[i])
                                os.makedirs(out_dir, exist_ok=True)

                                out_path = os.path.join(out_dir, file)
                                np.save(out_path, padded_current)
                                logging.info(
                                    f"Saved current curve part to {out_path}"
                                )

                        except Exception as exc:
                            logging.info(
                                f"Failed to process current curve file {file_path}",
                                exc_info=exc,
                            )
        logging.info("Finished cutting .npy files")

    def run_40sec_video_cutter(self) -> None:
        """
        Recursively searches through folders starting from the selected directory,
        and cuts each multi-frame .tif file into 8 fragments.
        Each fragment is saved to one of 8 hardcoded subdirectories in the same folder. (These subdirectories are made while doing this)
        """

        self.updateProgressTIFCutter(0)

        # Hardcoded 8 directory names
        dir_names = [
            "5Hz square",
            "Regular Photocurrent",
            "Incremental 5Hz square",
            "1Hz square",
            "1Hz ramp",
            "Spectral photocurrent",
            "Spectral 5Hz square",
            "Incremental photocurrent",
        ]

        root_dir: str = str(self.video_dirtxtbox.text())

        if not os.path.isdir(root_dir):
            logging.info("Selected path is not a directory.")
            return

        tif_files = []
        for folder, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith((".tif", ".btf")):
                    full_path = os.path.join(folder, file)
                    try:
                        with TiffFile(full_path) as tif:
                            if len(tif.pages) >= 5001:
                                tif_files.append(full_path)
                    except Exception as exc:
                        logging.info(
                            f"Skipping unreadable file {full_path}",
                            exc_info=exc,
                        )

        total_files = len(tif_files)

        if total_files == 0:
            logging.info("No .tif files found.")
            return

        for file_index, path in enumerate(tif_files):
            try:
                with TiffFile(path) as tif:
                    num_frames = len(tif.pages)

                    if num_frames < 2:
                        logging.info(
                            f"Skipping file with only {num_frames} frame(s): {path}"
                        )
                        continue

                    frames_per_cut = math.ceil(num_frames / 8)
                    base_dir = os.path.dirname(path)

                    for i in range(8):
                        end = num_frames - (i * frames_per_cut)
                        start = max(0, end - frames_per_cut)
                        frames = tif.asarray(key=range(start, end))

                        cut_dir = os.path.join(
                            base_dir, dir_names[7 - i]
                        )  # reverse order
                        os.makedirs(cut_dir, exist_ok=True)

                        out_path = os.path.join(cut_dir, "500 fps.TIF")
                        io.imsave(out_path, frames)
                    progress = round((file_index + 1) / total_files * 100)
                    self.updateProgressTIFCutter(progress)
                    logging.info(f"Finished cutting: {path}")

            except Exception as exc:
                logging.error(f"Error processing {path}", exc_info=exc)

            progress = round((file_index + 1) / total_files * 100)
            self.updateProgressTIFCutter(progress)

    def run_40sec_npy_splitter(self) -> None:
        """
        Processes .npy files in the selected directory:
        - Finds one current file (Ip*.npy) and one waveform file (*__Wavefroms_sr_50000.npy)
        - Splits them into 8 segments using hardcoded ranges
        - Saves the segments into 8 subdirectories with fixed names
        """

        self.updateProgressTIFCutter(0)

        """
        current_ranges = [
            (5, 250004),
            (250005, 500004),
            (500005, 750004),
            (750005, 1000004),
            (1000007, 1250006),
            (1250007, 1500006),
            (1500007, 1750004)
        ]

        waveform_ranges = [
            (0, 249999),
            (250000, 499999),
            (500000, 749999),
            (750000, 999999),
            (1000002, 1250001),
            (1250002, 1500001),
            (1500002, 1749999)
        ]
        """

        current_ranges = [
            (5, 250004),
            (250005, 500004),
            (500005, 750004),
            (750005, 1000004),
            (1000005, 1250004),
            (1250005, 1500004),
            (1500005, 1750004),
            (1750005, 2000004),
        ]

        waveform_ranges = [
            (0, 249999),
            (250000, 499999),
            (500000, 749999),
            (750000, 999999),
            (1000000, 1249999),
            (1250000, 1499999),
            (1500000, 1749999),
            (1750000, 1999999),
        ]

        dir_names = [
            "5Hz square",
            "Regular Photocurrent",
            "Incremental 5Hz square",
            "1Hz square",
            "1Hz ramp",
            "Spectral photocurrent",
            "Spectral 5Hz square",
            "Incremental photocurrent",
        ]

        root_dir = str(self.video_dirtxtbox.text())

        if not os.path.isdir(root_dir):
            logging.info("Selected path is not a directory.")
            return

        for folder, _, files in os.walk(root_dir):
            if any(folder.endswith(name) for name in dir_names):
                continue
            for file in files:
                file_path = os.path.join(folder, file)

                if file.endswith(".npy"):
                    if "__Wavefroms_sr_50000" in file:
                        try:
                            waveform_data = np.load(
                                file_path, allow_pickle=True
                            )
                            assert (
                                waveform_data.shape == (8,)
                                and waveform_data.dtype == object
                            )

                            for i, (start, end) in enumerate(waveform_ranges):
                                length = end - start + 1

                                # Build dtype for this slice
                                padded_length = length + 1 + 1
                                dtype = [
                                    ("Waveform", "<f8", (padded_length,)),
                                    ("Sepcification", "<U20"),
                                ]

                                waveform_list = []

                                for j, item in enumerate(waveform_data):
                                    wave = item["Waveform"][
                                        start : end + 1
                                    ]  # Access the waveform (first element of the tuple)
                                    padded_wave = np.concatenate(
                                        [
                                            np.full(1, wave[0]),
                                            wave,
                                            np.full(1, wave[-1]),
                                        ]
                                    )

                                    spec = item["Sepcification"]
                                    dtype = [
                                        (
                                            "Waveform",
                                            padded_wave.dtype,
                                            padded_wave.shape,
                                        ),
                                        ("Sepcification", "<U20"),
                                    ]
                                    record = np.array(
                                        (padded_wave, spec), dtype=dtype
                                    )[
                                        ()
                                    ]  # [()] unpacks to np.void
                                    waveform_list.append(record)

                                out_dir = os.path.join(folder, dir_names[i])
                                os.makedirs(out_dir, exist_ok=True)

                                out_path = os.path.join(out_dir, file)
                                waveform_array = np.array(
                                    waveform_list, dtype=object
                                )
                                np.save(out_path, waveform_array)
                                logging.info(
                                    f"Saved waveform part to {out_path}"
                                )

                        except Exception as exc:
                            logging.info(
                                f"Failed to process waveform file {file_path}",
                                exc_info=exc,
                            )

                    elif file.startswith("Ip") and file.endswith(".npy"):
                        try:
                            current_data = np.load(
                                file_path, allow_pickle=True
                            )
                            assert (
                                current_data.ndim == 1
                                and current_data.dtype == np.float64
                            )

                            for i, (start, end) in enumerate(current_ranges):
                                custom_padding_start = np.array(
                                    [
                                        5.00000000e04,
                                        -1.51955698e-03,
                                        1.61014993e-04,
                                        -5.23595314e-15,
                                        3.38772592e-18,
                                        0,
                                    ],
                                    dtype=np.float64,
                                )

                                padded_current = np.concatenate(
                                    [
                                        custom_padding_start,
                                        current_data[start : end + 1],
                                        np.zeros(1, dtype=np.float64),
                                    ]
                                )

                                out_dir = os.path.join(folder, dir_names[i])
                                os.makedirs(out_dir, exist_ok=True)

                                out_path = os.path.join(out_dir, file)
                                np.save(out_path, padded_current)
                                logging.info(
                                    f"Saved current curve part to {out_path}"
                                )

                        except Exception as exc:
                            logging.info(
                                f"Failed to process current curve file {file_path}",
                                exc_info=exc,
                            )
        logging.info("Finished cutting .npy files")

    def run_5sec_spectral5hz_video_cutter(self) -> None:
        """
        Recursively searches through folders starting from the selected directory,
        and cuts each multi-frame .tif file into 8 fragments.
        Each fragment is saved to one of 8 hardcoded subdirectories in the same folder. (These subdirectories are made while doing this)
        """

        self.updateProgressTIFCutter(0)

        # Hardcoded 5 directory names
        dir_names = [
            "532nm 5Hz square",
            "488nm 5Hz square",
            "532nm and 639nm 5Hz square",
            "488nm and 639nm 5Hz square",
            "488nm and 532nm 5Hz square",
        ]

        root_dir: str = str(self.video_dirtxtbox.text())

        if not os.path.isdir(root_dir):
            logging.info("Selected path is not a directory.")
            return

        tif_files = []
        for folder, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith((".tif", ".btf")):
                    full_path = os.path.join(folder, file)
                    try:
                        with TiffFile(full_path) as tif:
                            if len(tif.pages) >= 1001:
                                tif_files.append(full_path)
                    except Exception as exc:
                        logging.info(
                            f"Skipping unreadable file {full_path}",
                            exc_info=exc,
                        )

        total_files = len(tif_files)

        if total_files == 0:
            logging.info("No .tif files found.")
            return

        for file_index, path in enumerate(tif_files):
            try:
                with TiffFile(path) as tif:
                    num_frames = len(tif.pages)

                    if num_frames < 2:
                        logging.info(
                            f"Skipping file with only {num_frames} frame(s): {path}"
                        )
                        continue

                    frames_per_cut = math.ceil(num_frames / 5)
                    base_dir = os.path.dirname(path)

                    for i in range(5):
                        end = num_frames - (i * frames_per_cut)
                        start = max(0, end - frames_per_cut)
                        frames = tif.asarray(
                            key=range(start - 1, end - 1)
                        )  # Use this to use the first period as well, even though it might have an artifact: The first period is usually convoluted with photoactivation
                        # frames = tif.asarray(key=range(start+199, end-1)) # Use this to cut off the first 200 frames, because the first period is usually irregular.
                        cut_dir = os.path.join(
                            base_dir, dir_names[4 - i]
                        )  # reverse order
                        os.makedirs(cut_dir, exist_ok=True)

                        out_path = os.path.join(cut_dir, "1000 fps.TIF")
                        io.imsave(out_path, frames)
                    progress = round((file_index + 1) / total_files * 100)
                    self.updateProgressTIFCutter(progress)
                    logging.info(f"Finished cutting: {path}")

            except Exception as exc:
                logging.error(f"Error processing {path}", exc_info=exc)

            progress = round((file_index + 1) / total_files * 100)
            self.updateProgressTIFCutter(progress)

    def run_5sec_spectral5hz_npy_splitter(self) -> None:
        """
        Processes .npy files in the selected directory:
        - Finds one current file (Ip*.npy) and one waveform file (*__Wavefroms_sr_50000.npy)
        - Splits them into 8 segments using hardcoded ranges
        - Saves the segments into 8 subdirectories with fixed names
        """

        self.updateProgressTIFCutter(0)

        """
        # Use this to not cut off the first 200 ms
                current_ranges = [
                    (0, 49999),
                    (50000, 99999),
                    (100000, 149999),
                    (150000, 199999),
                    (200000, 249999)
                ]

                waveform_ranges = [
                    (0, 49999),
                    (50000, 99999),
                    (100000, 149999),
                    (150000, 199999),
                    (200000, 249999)
                ]

        """
        current_ranges = [
            (0, 49999),
            (50000, 99999),
            (100000, 149999),
            (150000, 199999),
            (200000, 249999),
        ]

        waveform_ranges = [
            (0, 49999),
            (50000, 99999),
            (100000, 149999),
            (150000, 199999),
            (200000, 249999),
        ]

        dir_names = [
            "532nm 5Hz square",
            "488nm 5Hz square",
            "532nm and 639nm 5Hz square",
            "488nm and 639nm 5Hz square",
            "488nm and 532nm 5Hz square",
        ]

        root_dir = str(self.video_dirtxtbox.text())

        if not os.path.isdir(root_dir):
            logging.info("Selected path is not a directory.")
            return

        for folder, _, files in os.walk(root_dir):
            if any(folder.endswith(name) for name in dir_names):
                continue
            for file in files:
                file_path = os.path.join(folder, file)

                if file.endswith(".npy"):
                    if "__Wavefroms_sr_50000" in file:
                        try:
                            waveform_data = np.load(
                                file_path, allow_pickle=True
                            )
                            assert (
                                waveform_data.shape == (8,)
                                and waveform_data.dtype == object
                            )

                            for i, (start, end) in enumerate(waveform_ranges):
                                length = end - start + 1

                                # Build dtype for this slice
                                padded_length = length + 1 + 1
                                dtype = [
                                    ("Waveform", "<f8", (padded_length,)),
                                    ("Sepcification", "<U20"),
                                ]

                                waveform_list = []

                                for j, item in enumerate(waveform_data):
                                    wave = item["Waveform"][
                                        start : end + 1
                                    ]  # Access the waveform (first element of the tuple)
                                    padded_wave = np.concatenate(
                                        [
                                            np.full(1, wave[0]),
                                            wave,
                                            np.full(1, wave[-1]),
                                        ]
                                    )

                                    spec = item["Sepcification"]
                                    dtype = [
                                        (
                                            "Waveform",
                                            padded_wave.dtype,
                                            padded_wave.shape,
                                        ),
                                        ("Sepcification", "<U20"),
                                    ]
                                    record = np.array(
                                        (padded_wave, spec), dtype=dtype
                                    )[
                                        ()
                                    ]  # [()] unpacks to np.void
                                    waveform_list.append(record)

                                out_dir = os.path.join(folder, dir_names[i])
                                os.makedirs(out_dir, exist_ok=True)

                                out_path = os.path.join(out_dir, file)
                                waveform_array = np.array(
                                    waveform_list, dtype=object
                                )
                                np.save(out_path, waveform_array)
                                logging.info(
                                    f"Saved waveform part to {out_path}"
                                )

                        except Exception as exc:
                            logging.info(
                                f"Failed to process waveform file {file_path}",
                                exc_info=exc,
                            )

                    elif file.startswith("Ip") and file.endswith(".npy"):
                        try:
                            current_data = np.load(
                                file_path, allow_pickle=True
                            )
                            assert (
                                current_data.ndim == 1
                                and current_data.dtype == np.float64
                            )

                            for i, (start, end) in enumerate(current_ranges):
                                custom_padding_start = np.array(
                                    [
                                        5.00000000e04,
                                        -1.51955698e-03,
                                        1.61014993e-04,
                                        -5.23595314e-15,
                                        3.38772592e-18,
                                        0,
                                    ],
                                    dtype=np.float64,
                                )

                                padded_current = np.concatenate(
                                    [
                                        custom_padding_start,
                                        current_data[start : end + 1],
                                        np.zeros(1, dtype=np.float64),
                                    ]
                                )

                                out_dir = os.path.join(folder, dir_names[i])
                                os.makedirs(out_dir, exist_ok=True)

                                out_path = os.path.join(out_dir, file)
                                np.save(out_path, padded_current)
                                logging.info(
                                    f"Saved current curve part to {out_path}"
                                )

                        except Exception as exc:
                            logging.info(
                                f"Failed to process current curve file {file_path}",
                                exc_info=exc,
                            )
        logging.info("Finished cutting .npy files")

    def exp_function(
        self, partial_trace: np.ndarray, a: float, b: float, c: float
    ) -> np.ndarray:
        """
        Exponential function used for fitting the signal part of the trace

        :param: partial_trace: np.ndarray: trace (not yet filtered)
        :param: a: float: scalefactor for function
        :param: b: float: rate of change for function
        :param: c: float: translation factor for function

        :returns: function: np.ndarray: values of function
        """
        return a * np.exp(-b * partial_trace) + c

    def apply_filter(self, trace: np.ndarray) -> np.ndarray:
        """
        Applies filter, note values for fs and cutoff seem to work but are not based on anything.
        :param: trace: np.ndarray: signal to be filtered
        :returns: np.ndarray: filtered signal
        """

        fs = 1 / 50000
        cutoff = fs / 500
        order = 5
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return filtfilt(b, a, trace)

    def show_ft(self, signal: np.ndarray):
        """
        function used for testing and visualizing fourier to determine cutoff freq and fs
        :param: signal: np.ndarray: (unfiltered) signal
        """
        fs = 500
        n = len(signal)
        # freq = np.fft.fftfreq(n, d=1 / fs)
        fft_signal = np.fft.fft(signal)

        f = np.fft.fftfreq(n, 1 / fs)
        # Plot the frequency spectrum
        plt.figure(figsize=(10, 6))
        logging.info(fft_signal)
        # plt.plot(np.abs(freq[[freq<=5]:n//2]), np.abs(fft_signal[[freq<=5]+1:n//2]))
        plt.plot(f, np.abs(fft_signal))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.title("Frequency Spectrum")
        # plt.show()

    def NIdaq_processor(
        self, trace: np.ndarray, baseline_index_length: int, i: int
    ) -> float:
        """
        For a single signal trace given the length of the baseline plots and filters the signal and fits a exp curve and determines the difference between intensity and baseline.
        Both baseline and signal are calculated by averaging of the filtered points within one standard deviation of the average of the last 1% of non-filtered points.

        :param: trace: np.ndarray: unfiltered signal
        :param: baseline_index_length: int: point where signal starts: baseline = [0,param)
        :param: i: int: itteration

        :returns: value: float: difference between signal and baseline
        """
        filtered_trace = self.apply_filter(trace)
        plt.plot(trace, label="unfiltered")
        plt.plot(filtered_trace, color="r", label="filtered")
        plt.title(f"filtered vs unfiltered with fit, iteration {i}")

        std_signal = np.std(filtered_trace[baseline_index_length:])
        signal_trace = filtered_trace[baseline_index_length:]
        # filtered_signal_trace = signal_trace[np.abs(signal_trace - signal_trace[-1])<=std_signal]
        filtered_signal_trace = signal_trace[
            np.abs(
                signal_trace
                - np.mean(
                    signal_trace[
                        int(np.shape(signal_trace)[0] * 0.9) : np.shape(
                            signal_trace
                        )[0]
                    ]
                )
            )
            <= std_signal
        ]  # @@ was 0.99
        filtered_signal_int = np.mean(filtered_signal_trace)

        std_baseline = np.std(filtered_trace[0:baseline_index_length])
        baseline_trace = filtered_trace[0:baseline_index_length]
        # filtered_baseline_trace = baseline_trace[np.abs(baseline_trace - baseline_trace[-1])<=std_baseline]
        filtered_baseline_trace = baseline_trace[
            np.abs(
                baseline_trace
                - np.mean(
                    baseline_trace[
                        int(np.shape(baseline_trace)[0] * 0.9) : np.shape(
                            baseline_trace
                        )[0]
                    ]
                )
            )
            <= std_baseline
        ]  # @@ was 0.99

        filtered_baseline_int = np.mean(filtered_baseline_trace)
        value = filtered_signal_int - filtered_baseline_int

        # fit exp
        x_data = np.linspace(
            baseline_index_length + 1,
            np.shape(filtered_trace)[0],
            np.shape(signal_trace)[0],
        )
        x_data_scaled = (x_data - np.min(x_data)) / (
            np.max(x_data) - np.min(x_data)
        )

        initial_guess = (0.005, 5, -0.18)
        try:
            popt, pcov = curve_fit(
                self.exp_function,
                x_data_scaled,
                trace[baseline_index_length:],
                p0=initial_guess,
            )
            plt.plot(
                x_data,
                self.exp_function(x_data_scaled, *popt),
                "--",
                color="black",
                label="fit: a=%5.3f, b=%5.3f, c=%5.3f" % tuple(popt),
            )
        except Exception as exc:
            logging.info(f"no fit possible for itteration {i}", exc_info=exc)

        plt.legend()
        # plt.show()

        # for plotting
        x_points_signal = np.linspace(
            baseline_index_length,
            np.shape(filtered_trace)[0],
            np.shape(filtered_trace)[0] - baseline_index_length,
        )
        x_points_baseline = np.linspace(
            0, baseline_index_length, baseline_index_length
        )
        x_points_baseline = x_points_baseline[
            np.abs(
                baseline_trace
                - np.mean(
                    baseline_trace[
                        int(np.shape(baseline_trace)[0] * 0.9) : np.shape(
                            baseline_trace
                        )[0]
                    ]
                )
            )
            <= std_baseline
        ]  # @@ was 0.99
        x_points_signal = x_points_signal[
            np.abs(
                signal_trace
                - np.mean(
                    signal_trace[
                        int(np.shape(signal_trace)[0] * 0.9) : np.shape(
                            signal_trace
                        )[0]
                    ]
                )
            )
            <= std_signal
        ]  # @@ was 0.99

        # Turned off, can be used for tuning the filter.
        # self.show_ft(filtered_trace)

        plt.plot(filtered_trace, zorder=1, label="filtered signal")
        plt.scatter(
            x_points_signal,
            filtered_signal_trace,
            color="g",
            zorder=2,
            marker="x",
            label="signal",
        )
        plt.scatter(
            x_points_baseline,
            filtered_baseline_trace,
            color="r",
            zorder=2,
            marker="x",
            label="baseline",
        )
        plt.title(f"iteration: {i}")
        plt.legend()
        # plt.show()

        return value

    def run_multi_NIdaq(self) -> list:
        """
        Parent function from whole trace creates small traces, selects baseline and signal, calculates difference and fits an exp function when possible
        """
        path = self.NIdaq_dir_variable
        signal_time = self.signal_length.value()
        baseline_time = self.baseline_length.value()
        end_time = self.cut_off_end.value()
        start_time = self.cut_off_start.value()
        sampling_rate = self.multi_waveform_samplingrate_box.value()

        intensity = []

        start_index = int(start_time * sampling_rate / 1000)

        end_index = -1
        if end_time != 0:
            end_index = int(end_time * sampling_rate / 1000)

        signal_index_length = signal_time * sampling_rate / 1000
        baseline_index_length = baseline_time * sampling_rate / 1000

        trace = np.load(path)

        iteration_index_length = signal_index_length + baseline_index_length
        logging.info(
            f"signal and baseline index: {signal_index_length}, {baseline_index_length}"
        )
        num_of_i = int(
            np.shape(trace[start_index:end_index])[0] / iteration_index_length
        )
        logging.info(start_index, end_index)
        chopped_trace = trace[start_index:end_index]

        # required for some reason... python doesn't want to python
        iteration_index_length = int(iteration_index_length)
        baseline_index_length = int(baseline_index_length)
        for i in range(num_of_i + 1):
            trace_iteration = chopped_trace[
                i * iteration_index_length : (i + 1) * iteration_index_length
            ]
            if np.shape(trace_iteration)[0] != 0:
                value = self.NIdaq_processor(
                    trace_iteration, baseline_index_length, i
                )
                # print(f'value: {value}')
                intensity.append(value)

            self.updateProgress_multiNIdaq(int(i / num_of_i * 100))

        logging.info(
            f" intensities calculated as list (signal-baseline): {intensity}"
        )

        return intensity

    def arctan(
        self, x: np.ndarray, a: float, b: float, c: float, d: float
    ) -> np.ndarray:
        return a * np.arctan(b * x + c) + d

    def tan(
        self, x: np.ndarray, a: float, b: float, c: float, d: float
    ) -> np.ndarray:
        return a * np.tan(b * x + c) + d

    def arctan_x(self, x, a, b, c):
        return a * b / ((b * x + c) ** 2 + 1)

    def zeros_of_arctan_xx(self, b):
        return (1 / (np.sqrt(3) * b), -1 / (np.sqrt(3) * b))

    def numerical_method(self, arr):
        first_derivative = np.diff(arr)
        second_derivative = np.diff(first_derivative)
        return first_derivative, second_derivative

    def ramp_filtering(self, trace):
        # kernel_size = 25
        window_length = 51
        polyorder = 3

        # filtered_trace = medfilt(trace, kernel_size=kernel_size)
        filtered_trace = savgol_filter(trace, window_length, polyorder)

        return filtered_trace

    def average_filtering(self, trace, window_size=20):

        filtered_trace = []
        for i in range(np.shape(trace)[0] - window_size):
            filtered_trace.append(np.average(trace[i : i + window_size]))

        return np.array(filtered_trace)

    def arctan_x_alg(self, x, c, index_of_interest, max_val):
        """
        extreme value = (index_of_interest, max_val)

        max_val = (ab)/[(b*index_of_interest + c)^2 + 1)]
        index_of_interest = -c/b => max_val = ab/[(b*-c/b+c)^2+1] = ab
        index_of_interest = -ca/max_val

        => f(x, c) = max_val/[(-c * x / index_of_interest + c)^2 + 1]
        """
        return max_val / ((-c * x / index_of_interest + c) ** 2 + 1)

    # CI feature still has to be implemented (current version is wrong, keep 0) could now lead to trying to acces points outside the indexed range
    def ramp_calculater(self, swing, CI=0) -> dict:
        ramp_results = {}

        #  plt.plot(swing, label='unfiltered')
        # swing = self.ramp_filtering(swing)
        filtered_swing = self.average_filtering(swing)

        first_der, second_der = self.numerical_method(filtered_swing)
        plt.plot(first_der, label="first derivative")
        plt.plot(second_der, label="second derivative")
        plt.title("Numerical derivatives of filtered Fluorescence (dF/dt)")
        plt.legend()
        # plt.show()

        first_der_filt, second_der_filt = self.average_filtering(
            first_der, window_size=self.ramp_avg_filt_kernel_size.value()
        ), self.average_filtering(
            second_der, window_size=self.ramp_avg_filt_kernel_size.value()
        )
        plt.plot(first_der_filt, label="first derivative filt")
        plt.plot(second_der_filt, label="second derivative filt")
        plt.title("Numerical filtered derivatives of Fluorescence (dF/dt)")

        # plt.show()

        index_of_interest = np.argmax(np.abs(first_der_filt))
        max_val = first_der_filt[index_of_interest]
        try:
            x_data = np.linspace(
                0, np.shape(first_der_filt)[0], np.shape(first_der_filt)[0]
            )
            x_data_scaled = (x_data - np.min(filtered_swing)) / (
                np.max(filtered_swing) - np.min(filtered_swing)
            )

            # index_of_interest_scaled = (
            #     index_of_interest - np.min(filtered_swing)
            # ) / (np.max(filtered_swing) - np.min(filtered_swing))

            plt.plot(x_data, first_der_filt, label="first der filt")

            popt, pcov = curve_fit(
                self.arctan_x, x_data_scaled, first_der_filt
            )

            a, b, c = popt
            # print(pcov)
            # a_std = float(pcov[0, 0])
            # b_std = float(pcov[1, 1])
            # c_std = float(pcov[2, 2])

            # plt.plot(x_data, self.arctan_x(x_data_scaled, *popt), '--', color='black', label=f'free fit: a={a:.{max(len(str(a_std)) - len(str(a_std)) - 2, 1)}}±{a_std:.{max(len(str(a_std)) - len(str(a_std)) - 2, 1)}}, b={b:.{max(len(str(b_std)) - len(str(b_std)) - 2, 1)}}±{b_std:.{max(len(str(b_std)) - len(str(b_std)) - 2, 1)}}, c={c:.{max(len(str(c_std)) - len(str(c_std)) - 2, 1)}}±{c_std:.{max(len(str(c_std)) - len(str(c_std)) - 2, 1)}}')
            # plt.plot(x_data, self.arctan_x(x_data_scaled, *popt), '--', color='black', label='free fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

            # for CI (to be fixed)
            # popt_alg, pcov_alg = curve_fit(lambda x, c: self.arctan_x_alg(x, c, index_of_interest, max_val), x_data[int(index_of_interest - CI/2*np.shape(x_data)[0]):int(index_of_interest + CI/2*np.shape(x_data)[0])], first_der_filt[int(index_of_interest - CI/2*np.shape(x_data)[0]):int(index_of_interest + CI/2*np.shape(x_data)[0])])
            popt_alg, pcov_alg = curve_fit(
                lambda x, c: self.arctan_x_alg(
                    x, c, index_of_interest, max_val
                ),
                x_data,
                first_der_filt,
            )

            c_alg = float(popt_alg[0])
            b_alg = float(-c_alg * index_of_interest)
            a_alg = float(max_val / b_alg)

            """
            b_alg_std = |partial b/ partial c| c_alg_std = |-index_of_interest | c_alg_std
            a_alg_std = |a| b_alg_std / |b| (error propagation formula)
            """

            c_alg_std = float(np.sqrt(pcov_alg[0]))
            b_alg_std = float(index_of_interest * c_alg_std)
            a_alg_std = float(np.abs(a_alg) * b_alg_std / np.abs(b_alg))

            logging.info(
                f"algebra fit: a,b,c = {a_alg}, {b_alg}, {c_alg} pm a_std, b_std, c_std = {a_alg_std},{b_alg_std},{c_alg_std} fit based on range size: {CI} most center points"
            )

            plt.plot(
                x_data,
                self.arctan_x_alg(
                    x_data, *popt_alg, index_of_interest, max_val
                ),
                "--",
                color="orange",
                label=f"algebra fit: a={a_alg:.{max(len(str(a_alg_std)) - len(str(a_alg_std)) - 2, 1)}}±{a_alg_std:.{max(len(str(a_alg_std)) - len(str(a_alg_std)) - 2, 1)}}, b={b_alg:.{max(len(str(b_alg_std)) - len(str(b_alg_std)) - 2, 1)}}±{b_alg_std:.{max(len(str(b_alg_std)) - len(str(b_alg_std)) - 2, 1)}}, c={c_alg:.{max(len(str(c_alg_std)) - len(str(c_alg_std)) - 2, 1)}}±{c_alg_std:.{max(len(str(c_alg_std)) - len(str(c_alg_std)) - 2, 1)}}",
            )
            # print(index_of_interest, max_val, popt_alg)

        except Exception as exc:
            logging.error("no fit possible, error", exc_info=exc)
        plt.legend()
        # plt.show()

        ramp_results["filtered_swing"] = filtered_swing
        ramp_results["index_of_interest"] = index_of_interest
        ramp_results["algebra_fit_abc"] = (a_alg, b_alg, c_alg)
        ramp_results["algebra_fit_std"] = (a_alg_std, b_alg_std, c_alg_std)

        return ramp_results

    # do not use
    def calc_avg_period(self, full_trace: np.ndarray) -> np.ndarray:
        freq = self.ramp_freq.value()
        samplingrate = self.ramp_fps.value()
        period_size = 1 / freq * samplingrate
        num_of_periods = np.shape(full_trace)[0] / period_size

        reshaped_trace = full_trace.reshape(
            int(num_of_periods), int(period_size)
        )
        avg_single_period = reshaped_trace.mean(axis=0)
        return avg_single_period

    def run_ramp_analysis(self) -> None:
        ramp_path: str = str(self.ramp_dir_variable)

        fluorecence: np.array = np.load(ramp_path)

        # fluorecence = self.calc_avg_period(fluorecence)

        plt.plot(fluorecence)
        plt.title("selected trace")
        # plt.show()

        # fluorecence = self.apply_filter(fluorecence)
        # plt.plot(fluorecence)
        # plt.title('filtered selected trace')
        # plt.show()
        # fluorecence = self.calc_avg_period(fluorecence)

        # assumes only one turn eg already averaged ramp
        turn_index = int(np.shape(fluorecence)[0] / 2)

        upswing = fluorecence[:turn_index]
        downswing = fluorecence[turn_index:]

        plt.plot(upswing, color="red", label="upswing")
        plt.plot(downswing, color="green", label="downswing")
        plt.legend()
        # plt.show()

        upswing_ramp_results = self.ramp_calculater(upswing)
        downswing_ramp_results = self.ramp_calculater(downswing)

        filt_swing = np.concatenate(
            (
                upswing_ramp_results["filtered_swing"],
                downswing_ramp_results["filtered_swing"],
            )
        )

        plt.plot(filt_swing, label="filtered swing")
        plt.plot(fluorecence, label="unfiltered swing")
        # print(np.shape(filt_upswing)[0])
        # val = np.shape(filt_upswing)[0]
        plt.plot(
            upswing_ramp_results["index_of_interest"],
            filt_swing[upswing_ramp_results["index_of_interest"]],
            "x",
        )
        plt.plot(
            np.shape(upswing_ramp_results["filtered_swing"])[0]
            + downswing_ramp_results["index_of_interest"],
            filt_swing[
                np.shape(upswing_ramp_results["filtered_swing"])[0]
                + downswing_ramp_results["index_of_interest"],
            ],
            "x",
        )
        plt.legend()
        # plt.show()

        kernel_size = self.ramp_avg_filt_kernel_size.value()
        down_volt = self.ramp_volt_min.value()
        up_volt = self.ramp_volt_max.value()

        upswing_volt = np.linspace(
            down_volt, up_volt, np.shape(fluorecence)[0]
        )
        upswing_volt_corrected = upswing_volt[:-kernel_size]
        upswing_volt_point = upswing_volt_corrected[
            upswing_ramp_results["index_of_interest"]
        ]

        downswing_volt = np.linspace(
            up_volt, down_volt, np.shape(fluorecence)[0]
        )
        # NOTE might be using wrong variable by accident
        downswing_volt_corrected = upswing_volt[:-kernel_size]
        downswing_volt_point = downswing_volt_corrected[
            downswing_ramp_results["index_of_interest"]
        ]

        logging.info(upswing_volt_point, downswing_volt_point)

    # - End of Ian's functions -

    def getfile(self):
        self.main_directory = str(QtWidgets.QFileDialog.getExistingDirectory())
        self.textbox_directory_name.setText(self.main_directory)

        if self.switch_Vp_or_camtrace.currentIndex() == 0:
            for file in os.listdir(self.main_directory):
                # For Labview generated data.
                if file.endswith(".tif") or file.endswith(".TIF"):
                    self.fileName = self.main_directory + "/" + file
                    logging.info(self.fileName)

            self.start_analysis()

        elif self.switch_Vp_or_camtrace.currentIndex() == 2:
            # For photocurrent analysis
            ProcessImage.PhotoCurrent(self.main_directory)

    def getfile_automated(self, path):
        self.main_directory = os.path.split(path)[0]
        logging.info(
            "Ian's automated function working on directory: ",
            self.main_directory,
        )
        self.textbox_directory_name.setText(self.main_directory)

        if self.switch_Vp_or_camtrace.currentIndex() == 0:

            for file in os.listdir(self.main_directory):
                # For Labview generated data.
                if file.endswith(".tif") or file.endswith(".TIF"):
                    self.fileName = self.main_directory + "/" + file
                    logging.info(self.fileName)

            self.start_analysis()

        elif self.switch_Vp_or_camtrace.currentIndex() == 2:
            # For photocurrent analysis
            ProcessImage.PhotoCurrent(self.main_directory)

    def start_analysis(self):
        """
        Getting the data folder, load the video

        Returns
        None.

        """
        if not os.path.exists(
            os.path.join(self.main_directory, "Patch analysis")
        ):
            # If the folder is not there, create the folder
            os.mkdir(os.path.join(self.main_directory, "Patch analysis"))

        logging.info("Loading data...")
        self.MessageToMainGUI("Loading data..." + "\n")

        t1 = threading.Thread(target=self.load_data_thread)
        self.button_load.setEnabled(False)
        t1.start()
        t1.join()

        self.MessageToMainGUI(
            "Video size: " + str(self.videostack.shape) + "\n"
        )
        self.roi_average.maxBounds = QRectF(
            0, 0, self.videostack.shape[2], self.videostack.shape[1]
        )
        self.roi_weighted.maxBounds = QRectF(
            0, 0, self.videostack.shape[2], self.videostack.shape[1]
        )
        logging.info(
            "============ Loading complete, ready to fire ============ "
        )
        self.MessageToMainGUI("=== Loading complete, ready to fire ===" + "\n")

        # Load wave files.
        self.loadcurve()

        # display electrical signals
        try:
            self.display_electrical_signals()
        except Exception as exc:
            logging.critical("caught exception", exc_info=exc)
            logging.info("No electrical recordings.")

        time.sleep(0.5)
        # Calculate the mean intensity of video, for background substraction.
        self.video_mean()

        logging.info("=========== Ready for analyse. =============")
        self.button_load.setEnabled(False)
        self.run_analysis_button.setEnabled(True)

    def finish_analysis(self):
        # @@ to prevent popups
        # plt.ion()
        t2 = threading.Thread(target=self.finish_analysis_thread)
        t2.start()
        t2.join()

        logging.info("============ Analysis done. ============")
        self.MessageToMainGUI("=== Analysis done. ===" + "\n")

        self.button_load.setEnabled(True)
        self.run_analysis_button.setEnabled(False)

    def load_data_thread(self):
        # Load tif video file.
        self.videostack = io.imread(self.fileName)
        logging.info(self.videostack.shape)

    def finish_analysis_thread(self):
        self.MessageToMainGUI("=== Analysis start.. ===" + "\n")
        # Calculate the background
        self.calculate_background_from_ROI_average()

        # Substract background
        self.substract_background()

        # calculate_weight
        self.calculate_weight()

        # display_weighted_trace
        self.display_weighted_trace()

        # Fit on weighted trace and sumarize the statistics.
        self.fit_on_trace()

        logging.info("============ Analysis done. ============")
        self.MessageToMainGUI("=== Analysis done. ===" + "\n")

        self.button_load.setEnabled(True)
        self.run_analysis_button.setEnabled(False)

    def ReceiveVideo(self, videosentin):
        self.videostack = videosentin
        logging.info(self.videostack.shape)
        self.MessageToMainGUI(
            "Video size: " + str(self.videostack.shape) + "\n"
        )
        self.roi_average.maxBounds = QRectF(
            0, 0, self.videostack.shape[2], self.videostack.shape[1]
        )
        self.roi_weighted.maxBounds = QRectF(
            0, 0, self.videostack.shape[2], self.videostack.shape[1]
        )
        logging.info("Loading complete, ready to fire")
        self.MessageToMainGUI("Loading complete, ready to fire" + "\n")

    def loadcurve(self):
        """
        Load the 1D array files, like voltage, current recordings or waveform information.

        By default the waveforms configured contain 4 extra samples at the front
        and 1 in the end, while the recorded waveforms contain 8 extra samples at
        the front and 1 in the end.

        Returns
        None.

        """
        found_correct_waveform_spelling = False
        for file in os.listdir(self.main_directory):
            # For Labview generated data.
            if file.endswith(".Ip"):
                self.Ipfilename = self.main_directory + "/" + file
                curvereadingobjective_i = ProcessImage.readbinaryfile(
                    self.Ipfilename
                )
                (
                    self.Ip,
                    self.samplingrate_display_curve,
                ) = curvereadingobjective_i.readbinarycurve()
                self.Ip = self.Ip[0 : len(self.Ip) - 2]

            elif file.endswith(".Vp"):
                self.Vpfilename = self.main_directory + "/" + file
                curvereadingobjective_V = ProcessImage.readbinaryfile(
                    self.Vpfilename
                )
                (
                    self.Vp,
                    self.samplingrate_display_curve,
                ) = curvereadingobjective_V.readbinarycurve()
                self.Vp = self.Vp[
                    0 : len(self.Vp) - 2
                ]  # Here -2 because there are two extra recording points in the recording file.

            # === Load configured waveform ===
            elif waveform_specification.is_waveform(file):
                if found_correct_waveform_spelling:
                    if waveform_specification.is_misspelled_wavefrom(file):
                        continue
                elif not waveform_specification.is_misspelled_wavefrom(file):
                    found_correct_waveform_spelling = True

                self.Waveform_filename_npy = self.main_directory + "/" + file
                # Read in configured waveforms
                configwave_wavenpfileName = self.Waveform_filename_npy
                self.configured_waveforms_container = (
                    waveform_specification.load(configwave_wavenpfileName)
                )

                # Get the sampling rate
                self.samplingrate_display_curve = (
                    waveform_specification.get_sample_rate(
                        configwave_wavenpfileName
                    )
                )
                logging.info(
                    "Waveforms_sampling rate: {}".format(
                        self.samplingrate_display_curve
                    )
                )

                # Load the configured patch voltage waveform
                for each_waveform in self.configured_waveforms_container:
                    if each_waveform["Specification"] == "patchAO":
                        self.configured_Vp = each_waveform["Waveform"]

                        logging.info(
                            "Length of configured_Vp: {}".format(
                                len(self.configured_Vp)
                            )
                        )

                        if len(self.configured_Vp) % 10 == 5:
                            # First 4 numbers and the last one in the recording
                            # are padding values outside the real camera recording.

                            # Extra camera triggers: False True True False | Real signals: True
                            # Extra patch voltage:   -0.7  -0.7 -0.7 -0.7  | Real signals: 0.3
                            self.configured_Vp = (
                                self.configured_Vp[4:][0:-1] / 10
                            )

                            logging.info(
                                "Length of cut Vp: {}".format(
                                    len(self.configured_Vp)
                                )
                            )

                        elif len(self.configured_Vp) % 10 == 2:
                            # The first number and the last one in the recording
                            # are padding values outside the real camera recording.
                            self.configured_Vp = (
                                self.configured_Vp[1:][0:-1] / 10
                            )

                            logging.info(
                                "Length of cut Vp: {}".format(
                                    len(self.configured_Vp)
                                )
                            )

            # For python generated data
            elif file.startswith("Vp"):
                self.Vpfilename_npy = self.main_directory + "/" + file
                curvereadingobjective_V = np.load(self.Vpfilename_npy)

                # In the recorded Vp trace, the 0 data is sampling rate,
                # 1 to 4 are NiDaq scaling coffecients,
                # 5 to 8 are extra samples for extra camera trigger,
                # The last one is padding 0 to reset NIDaq channels.
                self.Vp = curvereadingobjective_V[9:]
                self.samplingrate_display_curve = curvereadingobjective_V[0]
                self.Vp = self.Vp[0:-1]

                # This is from Fixed output from the patch amplifier, unfiltered, x10 already.
                # Ditched x10 voltage channel in amplifier from 22.12.2021
                # self.Vp = self.Vp/10

            elif file.startswith("Ip"):
                self.Ipfilename_npy = self.main_directory + "/" + file
                curvereadingobjective_I = np.load(self.Ipfilename_npy)
                # print("I raw: {}".format(curvereadingobjective_I[1000]))

                # In the recorded Vp trace, the 0 data is sampling rate,
                # 1 to 4 are NiDaq scaling coffecients,
                # 5 to 8 are extra samples for extra camera trigger,
                # The last one is padding 0 to reset NIDaq channels.
                self.Ip = curvereadingobjective_I[9:]
                self.Ip = self.Ip[0:-1]
                self.samplingrate_display_curve = curvereadingobjective_I[0]

    def getfile_background(self):
        self.fileName_background, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Single File",
            "M:/tnw/ist/do/projects/Neurophotonics/Brinkslab/Data",  # TODO hardcoded path
            "Image files (*.jpg *.tif)",
        )
        self.textbox_Background_filename.setText(self.fileName_background)

    def substract_background(self):
        """
        Substract the background from the original video.

        Returns
        None.

        """
        logging.info("Loading...")

        self.background_rolling_average_number = int(self.samplingrate_cam / 5)
        logging.info(
            "background_rolling_average_number is : {}".format(
                self.background_rolling_average_number
            )
        )

        unique, counts = np.unique(
            self.averageimage_ROI_mask, return_counts=True
        )
        count_dict = dict(zip(unique, counts))
        logging.info(f"number of 1 and 0: {count_dict}")

        self.background_trace = []

        for i in range(self.videostack.shape[0]):
            ROI_bg = (
                self.videostack[i][
                    self.roi_avg_coord_raw_start : self.roi_avg_coord_raw_start
                    + self.averageimage_ROI_mask.shape[0],
                    self.roi_avg_coord_col_start : self.roi_avg_coord_col_start
                    + self.averageimage_ROI_mask.shape[1],
                ]
                * self.averageimage_ROI_mask
            )

            # Sum of all pixel values and devided by non-zero pixel number
            bg_mean = np.sum(ROI_bg) / count_dict[1]

            self.background_trace.append(bg_mean)

        fig, ax0 = plt.subplots(figsize=(8.0, 5.8))
        fig.suptitle("Raw ROI background trace")
        plt.plot(self.cam_time_label, self.background_trace)
        ax0.set_xlabel("time(s)")
        ax0.set_ylabel("Pixel values")
        fig.savefig(
            os.path.join(
                self.main_directory,
                "Patch analysis//ROI raw background trace.png",
            ),
            dpi=1000,
        )
        # plt.show()

        # Use rolling average to smooth the background trace
        # self.background_trace = uniform_filter1d(uniform_filter1d
        # (self.background_trace, size=self.background_rolling_average_number), size=self.background_rolling_average_number*2)

        # Bi-exponential curve to fit the background
        def bg_func(t, a, t1, b, t2):
            return a * np.exp(-(t / t1)) + b * np.exp(-(t / t2))

        # Parameter bounds for the parameters in the bi-exponential function
        parameter_bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])

        # popt   = Optimal parameters for the curve fit
        # pcov   = The estimated covariance of popt
        popt, pcov = curve_fit(
            bg_func,
            self.cam_time_label,
            self.background_trace,
            bounds=parameter_bounds,
            maxfev=500000,
        )

        self.background_trace_smoothed = bg_func(self.cam_time_label, *popt)

        fig, ax1 = plt.subplots(figsize=(8.0, 5.8))
        fig.suptitle("Smoothed background trace")

        (p01,) = ax1.plot(
            self.cam_time_label,
            self.background_trace,
            color=(0, 0, 0.4),
            linestyle="None",
            marker="o",
            markersize=0.5,
            markerfacecolor=(0, 0, 0.9),
            label="Raw background",
        )
        (p02,) = ax1.plot(
            self.cam_time_label,
            self.background_trace_smoothed,
            color=(0.9, 0, 0),
            label="Smoothed background",
        )
        ax1.legend([p01, p02], ["Raw background", "Smoothed background"])
        ax1.set_xlabel("time(s)")
        ax1.set_ylabel("Pixel values")
        fig.savefig(
            os.path.join(
                self.main_directory,
                "Patch analysis//Smoothed background trace.png",
            ),
            dpi=1000,
        )
        # plt.show()

        # For each frame in video, substract the background
        container = np.empty(
            (
                self.videostack.shape[0],
                self.videostack.shape[1],
                self.videostack.shape[2],
            )
        )

        for i in range(self.videostack.shape[0]):
            raw_frame = self.videostack[i]

            background_mean = self.background_trace_smoothed[i]

            background_mean_2d = background_mean * np.ones(
                (self.videostack.shape[1], self.videostack.shape[2])
            )

            temp_diff = raw_frame - background_mean_2d
            temp_diff[temp_diff < 0] = 0

            container[i] = temp_diff
            self.videostack[i, :, :] = temp_diff

        logging.info("ROI background correction done.")

        saveBgSubstractedVideop = False
        # Save the file.
        if saveBgSubstractedVideop is True:
            with skimtiff.TiffWriter(
                os.path.join(
                    self.main_directory,
                    "Patch analysis//saveBgSubstractedVideo.tif",
                ),
                append=True,
            ) as tif:
                tif.save(self.videostack, compress=0)

        self.videostack = container
        # Show the background corrected trace.
        mean_camera_counts_backgroubd_substracted = []
        for i in range(self.videostack.shape[0]):
            mean_camera_counts_backgroubd_substracted.append(
                np.mean(self.videostack[i])
            )

        fig2, ax2 = plt.subplots(figsize=(8.0, 5.8))
        fig2.suptitle("Mean camera intensity after backgroubd substracted")
        plt.plot(
            self.cam_time_label, mean_camera_counts_backgroubd_substracted
        )
        ax2.set_xlabel("time(s)")
        ax2.set_ylabel("Pixel values")
        fig2.savefig(
            os.path.join(
                self.main_directory,
                "Patch analysis//Mean camera intensity after backgroubd substracted.png",
            ),
            dpi=1000,
        )
        # plt.show()

        # Updates the mean intensity
        self.imganalysis_averageimage = np.mean(self.videostack, axis=0)
        self.pw_averageimage.setImage(self.imganalysis_averageimage)

        fig3 = plt.figure(figsize=(8.0, 5.8))
        fig3.suptitle("Mean intensity")
        plt.imshow(self.imganalysis_averageimage)
        fig3.savefig(
            os.path.join(
                self.main_directory, "Patch analysis//Mean intensity.png"
            ),
            dpi=1000,
        )
        # plt.show()

    def display_electrical_signals(self):
        """
        Display the patch clamp electrode recored membrane potential and current signals.

        Returns
        None.

        """
        if self.switch_Vp_or_camtrace.currentIndex() == 0:
            self.patchcurrentlabel = (
                np.arange(len(self.Ip)) / self.samplingrate_display_curve
            )

            self.patchvoltagelabel = (
                np.arange(len(self.configured_Vp))
                / self.samplingrate_display_curve
            )

            self.electrical_signals_figure, (ax1, ax2) = plt.subplots(2, 1)
            # plt.title('Electrode recording')
            # Current here is already
            # Probe gain: low-100M ohem
            # [DAQ recording / 10**8 (voltage to current)]* 10**12 (A to pA) == pA
            ax1.plot(
                self.patchcurrentlabel,
                self.Ip * 10000,
                label="Current",
                color="b",
            )
            ax1.set_title("Electrode recording")
            ax1.set_xlabel("time(s)")
            ax1.set_ylabel("Current (pA)")
            # ax1.legend()

            # ax2 = self.electrical_signals_figure.add_subplot(212)
            # *1000: convert to mV; /10 is to correct for the *10 add on at patch amplifier.
            ax2.plot(
                self.patchvoltagelabel,
                self.configured_Vp * 1000,
                label="Set voltage",
                color="b",
            )
            # ax2.set_title('Voltage')
            ax2.set_xlabel("time(s)")
            ax2.set_ylabel("Set voltage(mV)")
            # ax2.legend()

            # plt.show()
            self.electrical_signals_figure.savefig(
                os.path.join(
                    self.main_directory,
                    "Patch analysis//Electrode recording.png",
                ),
                dpi=1000,
            )
        else:
            pass

    def video_mean(self):
        """
        Calculating the average 2d image from the video.

        Returns
        None.

        """
        self.imganalysis_averageimage = np.mean(self.videostack, axis=0)
        self.pw_averageimage.setImage(self.imganalysis_averageimage)
        # self.pw_averageimage.setLevels((50, 200))

        self.samplingrate_cam = self.Spincamsamplingrate.value()
        self.cam_time_label = (
            np.arange(self.videostack.shape[0]) / self.samplingrate_cam
        )

        fig = plt.figure(figsize=(8.0, 5.8))
        fig.suptitle("Mean intensity of raw video")
        plt.imshow(self.imganalysis_averageimage)
        fig.savefig(
            os.path.join(
                self.main_directory,
                "Patch analysis//Mean intensity of raw video.png",
            ),
            dpi=1000,
        )
        # plt.show()

        self.mean_camera_counts = []
        for i in range(self.videostack.shape[0]):
            self.mean_camera_counts.append(np.mean(self.videostack[i]))

        fig2, ax2 = plt.subplots(figsize=(8.0, 5.8))
        fig2.suptitle("Mean intensity trace of raw video")
        plt.plot(self.cam_time_label, self.mean_camera_counts)
        ax2.set_xlabel("time(s)")
        ax2.set_ylabel("Pixel values")
        fig2.savefig(
            os.path.join(
                self.main_directory,
                "Patch analysis//Mean intensity trace of raw video.png",
            ),
            dpi=1000,
        )
        # plt.show()

    def calculate_background_from_ROI_average(self):
        """
        Calculate the mean background value from the ROI selector.

        Returns
        None.

        """
        self.averageimage_imageitem = self.pw_averageimage.getImageItem()
        self.averageimage_ROI = self.roi_average.getArrayRegion(
            self.imganalysis_averageimage, self.averageimage_imageitem
        )
        self.averageimage_ROI_mask = np.where(self.averageimage_ROI > 0, 1, 0)

        # self.roi_average_pos = self.roi_average.pos()
        self.roi_average_Bounds = self.roi_average.parentBounds()
        self.roi_avg_coord_col_start = round(
            self.roi_average_Bounds.topLeft().x()
        )
        self.roi_avg_coord_col_end = round(
            self.roi_average_Bounds.bottomRight().x()
        )
        self.roi_avg_coord_raw_start = round(
            self.roi_average_Bounds.topLeft().y()
        )
        self.roi_avg_coord_raw_end = round(
            self.roi_average_Bounds.bottomRight().y()
        )

    def calculateweight(self):
        t2 = threading.Thread(target=self.calculate_weight)
        t2.start()

    def calculate_weight(self):
        """
        Calculate the pixels weights using correlation between the video and voltage reocrding.

        Returns
        None.

        """
        if self.switch_Vp_or_camtrace.currentIndex() == 0:
            self.samplingrate_cam = self.Spincamsamplingrate.value()
            self.downsample_ratio = int(
                self.samplingrate_display_curve / self.samplingrate_cam
            )

            logging.info(
                "Vp downsampling ratio: {}".format(self.downsample_ratio)
            )
            logging.info(
                "Sampling rate camera: {}".format(self.samplingrate_cam)
            )
            logging.info(
                "Sampling rate DAQ: {}".format(self.samplingrate_display_curve)
            )

            try:
                self.Vp_downsample = self.configured_Vp.reshape(
                    -1, self.downsample_ratio
                ).mean(axis=1)

                # self.Vp_downsample = self.Vp_downsample[0 : len(self.videostack)]
            except Exception as exc:
                logging.critical("caught exception", exc_info=exc)
                logging.info("Vp downsampling ratio is not an integer.")
                small_ratio = int(
                    np.floor(
                        self.samplingrate_display_curve / self.samplingrate_cam
                    )
                )
                resample_length = int(len(self.videostack) * small_ratio)
                self.Vp_downsample = signal.resample(
                    self.configured_Vp, resample_length
                )
                plt.figure()
                plt.plot(self.Vp_downsample)
                # plt.show()

                self.Vp_downsample = self.Vp_downsample.reshape(
                    -1, small_ratio
                ).mean(axis=1)

            # If there's a missing frame in camera video(probably in the beginning),
            # cut the Vp from the beginning
            self.Vp_downsample = self.Vp_downsample[
                len(self.Vp_downsample) - len(self.videostack) :
            ]

            logging.info(f"{self.videostack.shape} {self.Vp_downsample.shape}")

            (
                self.corrimage,
                self.weightimage,
                self.sigmaimage,
            ) = ProcessImage.extractV(
                self.videostack, self.Vp_downsample * 1000
            )
            # *1000: convert to mV; /10 is to correct for the *10 add on at patch amplifier(not anymore).

            self.pw_weightimage.setImage(self.weightimage)

        elif self.switch_Vp_or_camtrace.currentIndex() == 1:
            (
                self.corrimage,
                self.weightimage,
                self.sigmaimage,
            ) = ProcessImage.extractV(
                self.videostack, self.camsignalsum * 1000
            )

            self.pw_weightimage.setImage(self.weightimage)

        fig = plt.figure(figsize=(8.0, 5.8))
        fig.suptitle("Weighted pixels")
        plt.imshow(self.weightimage)
        fig.savefig(
            os.path.join(
                self.main_directory, "Patch analysis//Weighted pixel image.png"
            ),
            dpi=1000,
        )
        np.save(
            os.path.join(
                self.main_directory, "Patch analysis//Weighted pixel image.npy"
            ),
            self.weightimage,
        )
        # plt.show()

    def display_weighted_trace(self):
        """
        Display the mean sum weight and display frame by frame.

        Returns
        None.

        """
        self.videolength = len(self.videostack)

        # datv = squeeze(mean(mean(mov.*repmat(Wv./sum(Wv(:))*movsize(1)*movsize(2), [1 1 length(sig)]))));
        k = np.tile(
            self.weightimage
            / np.sum(self.weightimage)
            * self.videostack.shape[1]
            * self.videostack.shape[2],
            (self.videolength, 1, 1),
        )
        self.weighttrace_tobetime = self.videostack * k

        self.weight_trace_data = np.zeros(self.videolength)
        for i in range(self.videolength):
            self.weight_trace_data[i] = np.mean(self.weighttrace_tobetime[i])

        self.patch_camtrace_label_weighted = (
            np.arange(self.videolength) / self.samplingrate_cam
        )

        np.save(
            os.path.join(
                self.main_directory, "Patch analysis//Weighted_trace.npy"
            ),
            self.weight_trace_data,
        )

        fig, ax1 = plt.subplots(figsize=(8.0, 5.8))
        fig.suptitle("Weighted pixel trace")
        plt.plot(self.patch_camtrace_label_weighted, self.weight_trace_data)
        ax1.set_xlabel("time(s)")
        ax1.set_ylabel("Weighted trace(counts)")
        fig.savefig(
            os.path.join(
                self.main_directory, "Patch analysis//Weighted pixel trace.png"
            ),
            dpi=1000,
        )
        # plt.show()

    def fit_on_trace(self):
        """
        Using curve fit function to calculate all the statistics.

        Returns
        None.

        """
        fit = PatchAnalysis(
            weighted_trace=self.weight_trace_data,
            DAQ_waveform=self.configured_Vp,
            voltage_step_frequency=self.VoltageStepFrequencyQSpinBox.value(),
            camera_fps=self.samplingrate_cam,
            DAQ_sampling_rate=self.samplingrate_display_curve,
            main_directory=self.main_directory,
            rhodopsin=self.Construct_name.text(),
        )
        fit.Photobleach()
        fit.ExtractPeriod()
        fit.FitSinglePeriod()
        fit.ExtractSensitivity()
        fit.Statistics()

    def save_analyzed_image(self, catag):
        if catag == "weight_image":
            Localimg = Image.fromarray(
                self.weightimage
            )  # generate an image object
            Localimg.save(
                os.path.join(
                    self.savedirectory,
                    "Weight_"
                    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    + ".tif",
                )
            )  # save as tif

    def clearplots(self):
        self.button_load.setEnabled(True)
        self.run_analysis_button.setEnabled(False)

        self.pw_weightimage.clear()
        self.pw_averageimage.clear()

        self.videostack = None

    def MessageToMainGUI(self, text):
        self.MessageBack.emit(text)

    def send_DMD_mask_contour(self, contour_from_cellselection):
        self.Cellselection_DMD_mask_contour.emit(contour_from_cellselection)

    def get_single_waveform(self):
        """
        Display a single trace.

        Returns
        None.

        """
        (
            self.single_waveform_fileName,
            _,
        ) = QtWidgets.QFileDialog.getOpenFileName(self, "Single File")
        self.textbox_single_waveform_filename.setText(
            self.single_waveform_fileName
        )

        self.single_waveform = np.load(
            self.single_waveform_fileName, allow_pickle=True
        )

        wave_sampling_rate = self.waveform_samplingrate_box.value()

        time_axis = (
            np.arange(len(self.single_waveform[9:-1])) / wave_sampling_rate
        )

        try:
            if "Ip" in self.single_waveform_fileName:
                # If plotting the patch current
                self.Ip = self.single_waveform[9:-1]

                fig, ax = plt.subplots()
                plt.plot(time_axis, self.Ip * 10000)
                ax.set_title("Patch current")
                ax.set_ylabel("Current (pA)")
                ax.set_xlabel("time(s)")
            elif "Vp" in self.single_waveform_fileName:
                # If plotting the patch voltage
                self.Vp = self.single_waveform[9:-1]

                fig, ax = plt.subplots()
                plt.plot(time_axis, self.Vp * 1000)
                ax.set_title("Patch voltage")
                ax.set_ylabel("Membrane potential (mV)")
                ax.set_xlabel("time(s)")

                logging.info(
                    "For Vp recored earlier than 22.12.2021, pls devided by 10 as the amplifier channel multipliesby 10 by default."
                )
            elif "PMT" in self.single_waveform_fileName:
                self.PMT_recording = self.single_waveform[9:-1]

                fig, ax = plt.subplots()
                plt.plot(time_axis, self.PMT_recording)
                ax.set_title("PMT voltage")
                ax.set_ylabel("Volt (V)")
                ax.set_xlabel("time(s)")

            else:
                plt.figure()
                plt.plot(self.single_waveform_fileName)

            # plt.show()
        except Exception as exc:
            logging.critical("caught exception", exc_info=exc)


# %%
class PlotAnalysisGUI(QWidget):
    waveforms_generated = pyqtSignal(object, object, list, int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # === Initiating patchclamp class ===
        # === GUI ===
        self.setMinimumSize(200, 200)
        self.setWindowTitle("Plot display")
        self.layout = QGridLayout(self)

        pmtimageContainer = QGroupBox("Read-in")
        self.pmtimageLayout = QGridLayout()

        self.checkboxWaveform = QCheckBox("Waveform")
        self.checkboxWaveform.setStyleSheet(
            'color:CadetBlue;font:bold "Times New Roman"'
        )
        self.checkboxWaveform.setChecked(True)
        self.layout.addWidget(self.checkboxWaveform, 0, 0)

        self.checkboxTrace = QCheckBox("Recorded trace")
        self.checkboxTrace.setStyleSheet(
            'color:CadetBlue;font:bold "Times New Roman"'
        )
        self.layout.addWidget(self.checkboxTrace, 1, 0)

        self.checkboxCam = QCheckBox("Cam trace")
        self.checkboxCam.setStyleSheet(
            'color:CadetBlue;font:bold "Times New Roman"'
        )

        self.Spincamsamplingrate = QSpinBox(self)
        self.Spincamsamplingrate.setMaximum(2000)
        self.Spincamsamplingrate.setValue(250)
        self.Spincamsamplingrate.setSingleStep(250)
        self.layout.addWidget(self.Spincamsamplingrate, 2, 2)
        self.layout.addWidget(QLabel("Camera FPS:"), 2, 1)

        self.layout.addWidget(self.checkboxCam, 2, 0)

        self.savedirectorytextbox = QtWidgets.QLineEdit(self)
        self.pmtimageLayout.addWidget(self.savedirectorytextbox, 1, 0)

        # self.v_directorytextbox = QtWidgets.QLineEdit(self)
        # self.pmtimageLayout.addWidget(self.v_directorytextbox, 2, 0)

        self.toolButtonOpenDialog = QtWidgets.QPushButton("Select folder")

        self.toolButtonOpenDialog.clicked.connect(self._open_file_dialog)

        self.pmtimageLayout.addWidget(self.toolButtonOpenDialog, 1, 1)

        self.toolButtonLoad = QtWidgets.QPushButton("Graph")

        self.toolButtonLoad.clicked.connect(self.show_graphy)
        self.pmtimageLayout.addWidget(self.toolButtonLoad, 1, 2)

        pmtimageContainer.setLayout(self.pmtimageLayout)
        self.layout.addWidget(pmtimageContainer, 3, 0, 1, 3)

    def _open_file_dialog(self):
        self.Nest_data_directory = str(
            QtWidgets.QFileDialog.getExistingDirectory()
        )
        self.savedirectorytextbox.setText(self.Nest_data_directory)

    def show_graphy(self):
        self.cam_trace_fluorescence_dictionary = {}
        self.cam_trace_fluorescence_filename_dictionary = {}
        self.region_file_name = []

        found_correct_waveform_spelling = False
        for file in os.listdir(self.Nest_data_directory):
            if waveform_specification.is_waveform(file):
                if found_correct_waveform_spelling:
                    if waveform_specification.is_misspelled_wavefrom(file):
                        continue
                elif not waveform_specification.is_misspelled_wavefrom(file):
                    found_correct_waveform_spelling = True

                self.wave_fileName = os.path.join(
                    self.Nest_data_directory, file
                )
            elif file.endswith("csv"):  # Quick dirty fix
                self.recorded_cam_fileName = os.path.join(
                    self.Nest_data_directory, file
                )

                self.samplingrate_cam = self.Spincamsamplingrate.value()
                self.cam_trace_time_label = np.array([])
                self.cam_trace_fluorescence_value = np.array([])

                with open(self.recorded_cam_fileName, newline="") as csvfile:
                    spamreader = csv.reader(
                        csvfile, delimiter=" ", quotechar="|"
                    )
                    for column in spamreader:
                        coords = column[0].split(",")
                        if coords[0] != "X":  # First row and column is 'x, y'
                            self.cam_trace_time_label = np.append(
                                self.cam_trace_time_label, int(coords[0])
                            )
                            self.cam_trace_fluorescence_value = np.append(
                                self.cam_trace_fluorescence_value,
                                float(coords[1]),
                            )
                self.cam_trace_fluorescence_dictionary[
                    "region_{0}".format(len(self.region_file_name) + 1)
                ] = self.cam_trace_fluorescence_value
                self.cam_trace_fluorescence_filename_dictionary[
                    "region_{0}".format(len(self.region_file_name) + 1)
                ] = file
                self.region_file_name.append(file)
            elif "Vp" in file:
                self.recorded_wave_fileName = os.path.join(
                    self.Nest_data_directory, file
                )

        # Read in configured waveforms
        configwave_wavenpfileName = self.wave_fileName
        temp_loaded_container = waveform_specification.load(
            configwave_wavenpfileName
        )

        Daq_sample_rate = waveform_specification.get_sample_rate(
            configwave_wavenpfileName
        )

        self.Checked_display_list = ["Waveform"]
        if self.checkboxTrace.isChecked():
            self.Checked_display_list = np.append(
                self.Checked_display_list, "Recorded_trace"
            )
        if self.checkboxCam.isChecked():
            self.Checked_display_list = np.append(
                self.Checked_display_list, "Cam_trace"
            )

        reference_length = len(temp_loaded_container[0]["Waveform"])
        xlabel_all = np.arange(reference_length) / Daq_sample_rate

        # === For patch perfusion ===
        if len(self.region_file_name) == 0:
            if len(self.Checked_display_list) == 2:
                figure, (ax1, ax2) = plt.subplots(2, 1)

            elif len(self.Checked_display_list) == 3:
                figure, (ax1, ax2, ax3) = plt.subplots(3, 1)

            for i in range(len(temp_loaded_container)):
                if temp_loaded_container[i]["Specification"] == "640AO":
                    pass
                elif temp_loaded_container[i]["Specification"] == "488AO":
                    ax1.plot(
                        xlabel_all,
                        temp_loaded_container[i]["Waveform"],
                        label="488AO",
                        color="b",
                    )
                elif (
                    temp_loaded_container[i]["Specification"] == "Perfusion_8"
                ):
                    ax1.plot(
                        xlabel_all,
                        temp_loaded_container[i]["Waveform"],
                        label="KCL",
                    )
                elif (
                    temp_loaded_container[i]["Specification"] == "Perfusion_7"
                ):
                    ax1.plot(
                        xlabel_all,
                        temp_loaded_container[i]["Waveform"],
                        label="EC",
                    )
                elif (
                    temp_loaded_container[i]["Specification"] == "Perfusion_2"
                ):
                    ax1.plot(
                        xlabel_all,
                        temp_loaded_container[i]["Waveform"],
                        label="Suction",
                    )
            ax1.set_title("Output waveforms")
            ax1.set_xlabel("time(s)")
            ax1.set_ylabel("Volt")
            ax1.legend()

            figure.tight_layout()
            # plt.show()  # TODO cleanup
        # === For plots with camera regions ===
        if len(self.region_file_name) != 0:
            for region_number in range(len(self.region_file_name)):
                if len(self.Checked_display_list) == 2:
                    figure, (ax1, ax2) = plt.subplots(2, 1)
                    logging.info(1111)
                elif len(self.Checked_display_list) == 3:
                    figure, (ax1, ax2, ax3) = plt.subplots(3, 1)

                for i in range(len(temp_loaded_container)):
                    if temp_loaded_container[i]["Specification"] == "640AO":
                        ax1.plot(
                            xlabel_all,
                            temp_loaded_container[i]["Waveform"],
                            label="640AO",
                            color="r",
                        )
                    elif temp_loaded_container[i]["Specification"] == "488AO":
                        ax1.plot(
                            xlabel_all,
                            temp_loaded_container[i]["Waveform"],
                            label="488AO",
                            color="b",
                        )
                    elif (
                        temp_loaded_container[i]["Specification"]
                        == "Perfusion_8"
                    ):
                        ax1.plot(
                            xlabel_all,
                            temp_loaded_container[i]["Waveform"],
                            label="KCL",
                        )
                    elif (
                        temp_loaded_container[i]["Specification"]
                        == "Perfusion_7"
                    ):
                        ax1.plot(
                            xlabel_all,
                            temp_loaded_container[i]["Waveform"],
                            label="EC",
                        )
                    elif (
                        temp_loaded_container[i]["Specification"]
                        == "Perfusion_2"
                    ):
                        ax1.plot(
                            xlabel_all,
                            temp_loaded_container[i]["Waveform"],
                            label="Suction",
                        )
                ax1.set_title("Output waveforms")
                ax1.set_xlabel("time(s)")
                ax1.set_ylabel("Volt")
                ax1.legend()

                if "Recorded_trace" in self.Checked_display_list:
                    # Read in recorded waves
                    Readin_fileName = self.recorded_wave_fileName

                    if (
                        "Vp" in os.path.split(Readin_fileName)[1]
                    ):  # See which channel is recorded
                        Vm = np.load(Readin_fileName, allow_pickle=True)
                        Vm = Vm[9:-1]  # first 5 are sampling rate, Daq coffs

                    ax2.set_xlabel("time(s)")
                    ax2.set_title("Recording")
                    ax2.set_ylabel("V (Vm*10)")
                    ax2.plot(xlabel_all, Vm, label="Vm")
                    ax2.legend()
                elif (
                    "Recorded_trace" not in self.Checked_display_list
                    and len(self.Checked_display_list) == 2
                ):
                    ax2.plot(
                        self.cam_trace_time_label / self.samplingrate_cam,
                        self.cam_trace_fluorescence_dictionary[
                            "region_{0}".format(region_number + 1)
                        ],
                        label="Fluorescence",
                    )
                    ax2.set_xlabel("time(s)")
                    ax2.set_title(
                        "ROI Fluorescence"
                        + " ("
                        + str(
                            self.cam_trace_fluorescence_filename_dictionary[
                                "region_{0}".format(region_number + 1)
                            ]
                        )
                        + ")"
                    )
                    ax2.set_ylabel("CamCounts")
                    ax2.legend()

                if len(self.Checked_display_list) == 3:
                    ax3.plot(
                        self.cam_trace_time_label / self.samplingrate_cam,
                        self.cam_trace_fluorescence_dictionary[
                            "region_{0}".format(region_number + 1)
                        ],
                        label="Fluorescence",
                    )
                    ax3.set_xlabel("time(s)")
                    ax3.set_title(
                        "ROI Fluorescence"
                        + " ("
                        + str(
                            self.cam_trace_fluorescence_filename_dictionary[
                                "region_{0}".format(region_number + 1)
                            ]
                        )
                        + ")"
                    )
                    ax3.set_ylabel("CamCounts")
                    ax3.legend()
                figure.tight_layout()
                # plt.show()


if __name__ == "__main__":

    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        pg.setConfigOptions(imageAxisOrder="row-major")
        mainwin = AnalysisWidgetUI()
        mainwin.show()
        app.exec_()

    run_app()
