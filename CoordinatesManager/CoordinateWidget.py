#!/usr/bin/env python3
# -*- coding: utf8 -*-
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:57:36 2020

@author: Izak de Heer
"""

import sys
import os
import json

# Ensure that the Widget can be run either independently or as part of Tupolev.
if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname + "/../")
    # os.chdir(os.getcwd()+'/')

# Backend

from CoordinatesManager.backend.readRegistrationImages import touchingCoordinateFinder
from CoordinatesManager.backend.polynomialTransformation import polynomialRegression
from CoordinatesManager.backend.Registrator import RegistrationThread
from CoordinatesManager import DMDActuator
from CoordinatesManager import ManualRegistration

from NIDAQ.DAQoperator import DAQmission
from NIDAQ.generalDaqerThread import execute_analog_readin_optional_digital_thread
from NIDAQ.wavegenerator import waveRecPic

from ImageAnalysis.ImageProcessing import ProcessImage
from HamamatsuCam.HamamatsuActuator import CamActuator
from GalvoWidget.pmt_thread import pmtimagingTest_contour

# UI
from CoordinatesManager.ui_widgets.adaptedQButtonGroupClass import adaptedQButtonGroup
from CoordinatesManager.ui_widgets.DrawingWidget import DrawingWidget
from CoordinatesManager.ui_widgets.SelectPointImageView import SelectPointImageView

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QWidget,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QVBoxLayout,
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
)

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from StylishQT import MySwitch, roundQGroupBox, SquareImageView

import pyqtgraph as pg
from pyqtgraph import QtGui

# Image processing
from skimage.draw import line, polygon2mask, polygon_perimeter
from skimage.morphology import binary_dilation
from PIL import Image

# General libraries
import threading
import sys
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt


class CoordinatesWidgetUI(QWidget):

    sig_start_registration = pyqtSignal()
    sig_finished_registration = pyqtSignal()
    sig_control_laser = pyqtSignal(str, float)
    sig_console_print = pyqtSignal(str)

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.backend = DMDWidget(self)
        self.main_application = parent
        self.init_gui()

    def closeEvent(self, event):
        self.backend.disconnectDMD()

        self.sig_control_laser.emit("640", 0)
        self.sig_control_laser.emit("532", 0)
        self.sig_control_laser.emit("488", 0)

        QtWidgets.QApplication.quit()
        event.accept()

    def init_gui(self):
        self.setWindowTitle("DMD management")
        # self.setMinimumSize(1000,800)
        # self.setMaximumSize(1000,800)

        self.dmdWidgetLayout = QGridLayout()
        self.setLayout(self.dmdWidgetLayout)

        self.dmdWidgetLayout.setColumnStretch(0, 5)
        self.dmdWidgetLayout.setColumnStretch(1, 5)
        self.dmdWidgetLayout.setColumnStretch(2, 1)

        if __name__ == "__main__":
            self.noticeBoard = QtWidgets.QTextEdit()
            self.noticeBoard.setFontItalic(True)
            self.noticeBoard.setPlaceholderText("Notice board from console.")
            self.noticeBoard.setMaximumHeight(100)
            self.setMinimumSize(1200, 1000)

            self.dmdWidgetLayout.addWidget(self.noticeBoard, 5, 0, 1, 5)

        # ----------------------  Stage control container --------------------

        self.stageContainer = roundQGroupBox("Manual registration tool")
        self.stageContainer.setMaximumHeight(150)
        self.stageContainerLayout = QGridLayout()
        self.stageContainer.setLayout(self.stageContainerLayout)
        self.stageRegisterButton = QPushButton("Open tool")
        self.stageRegisterButton.clicked.connect(self.backend.manual_registration)
        self.stageRegisterButton.setStyleSheet(
            "QPushButton {color:white;background-color: blue; border-style: outset;border-radius: 8px;border-width: 2px;font: bold 12px;padding: 6px}"
            "QPushButton:pressed {color:blue;background-color: darkblue; border-style: outset;border-radius: 8px;border-width: 2px;font: bold 12px;padding: 6px}"
            "QPushButton:hover:!pressed {color:blue;background-color: lightblue; border-style: outset;border-radius: 8px;border-width: 2px;font: bold 12px;padding: 6px}"
            "QPushButton:disabled: {color:blue;background-color: lightred; border-style: outset;border-radius: 8px;border-width: 2px;font: bold 12px;padding: 6px}"
        )

        self.stageContainerLayout.addWidget(self.stageRegisterButton, 0, 0)

        self.dmdWidgetLayout.addWidget(self.stageContainer, 0, 0)

        # ----------------------  Galvo scanning Container  -------------------

        self.galvosContainer = roundQGroupBox("Galvo's")
        self.galvosContainer.setMaximumHeight(150)
        self.galvosContainerLayout = QGridLayout()
        self.galvosContainer.setLayout(self.galvosContainerLayout)
        self.galvosRegisterButton = QPushButton("Register")
        self.galvosRegisterButton.setStyleSheet(
            "QPushButton {color:white;background-color: blue; border-style: outset;border-radius: 8px;border-width: 2px;font: bold 12px;padding: 6px}"
            "QPushButton:pressed {color:blue;background-color: darkblue; border-style: outset;border-radius: 8px;border-width: 2px;font: bold 12px;padding: 6px}"
            "QPushButton:hover:!pressed {color:blue;background-color: lightblue; border-style: outset;border-radius: 8px;border-width: 2px;font: bold 12px;padding: 6px}"
            "QPushButton:disabled: {color:blue;background-color: lightred; border-style: outset;border-radius: 8px;border-width: 2px;font: bold 12px;padding: 6px}"
        )

        self.galvosScanMaskButton = QPushButton("Scan mask")
        self.galvosScanMaskButton.setStyleSheet(
            "QPushButton {color:white;background-color: green; border-style: outset;border-radius: 8px;border-width: 2px;font: bold 12px;padding: 6px}"
            "QPushButton:pressed {color:green;background-color: darkgreen; border-style: outset;border-radius: 8px;border-width: 2px;font: bold 12px;padding: 6px}"
            "QPushButton:hover:!pressed {color:green;background-color: lightgreen; border-style: outset;border-radius: 8px;border-width: 2px;font: bold 12px;padding: 6px}"
        )

        self.galvosRegisterButton.clicked.connect(self.backend.galvos_register)
        self.galvosScanMaskButton.clicked.connect(self.backend.galvos_scan_mask)

        self.galvosScanStartSwitch = MySwitch(
            "STOP", "red", "START", "maroon", width=50
        )
        self.galvosScanStartSwitch.clicked.connect(self.backend.start_galvo_scan)

        self.galvosContainerLayout.addWidget(self.galvosRegisterButton, 0, 0)
        self.galvosContainerLayout.addWidget(self.galvosScanMaskButton, 1, 0)
        self.galvosContainerLayout.addWidget(self.galvosScanStartSwitch, 0, 1)
        self.dmdWidgetLayout.addWidget(self.galvosContainer, 0, 1, 1, 2)

        # ----------------------  DMD projection Container  ------------------

        dmdContainer = roundQGroupBox("DMD mask projection")
        dmdContainer.setMaximumHeight(150)
        self.dmdContainerLayout = QGridLayout()

        self.dmdConnectButton = MySwitch(
            "Disconnect", "red", "Connect", "maroon", width=70
        )
        self.dmdConnectButton.clicked.connect(self.backend.handle_dmd_connection)

        self.dmdIlluminateAllButton = QPushButton("Full illumination")
        self.dmdIlluminateAllButton.clicked.connect(self.backend.dmd_illuminate_all)

        self.dmdRegistrationButton = QPushButton("Register")
        self.dmdRegistrationButton.setStyleSheet(
            "QPushButton:disabled {color:white;background-color: grey; border-style: outset;border-radius: 8px;border-width: 2px;font: bold 12px;padding: 6px}"
            "QPushButton {color:white;background-color: blue; border-style: outset;border-radius: 8px;border-width: 2px;font: bold 12px;padding: 6px}"
            "QPushButton:pressed {color:blue;background-color: darkblue; border-style: outset;border-radius: 8px;border-width: 2px;font: bold 12px;padding: 6px}"
            "QPushButton:hover:!pressed {color:blue;background-color: lightblue; border-style: outset;border-radius: 8px;border-width: 2px;font: bold 12px;padding: 6px}"
        )

        self.dmdRegistrationButton.clicked.connect(self.select_lasers_popup)

        self.loadMaskButton = QPushButton("Load mask")
        self.loadMaskButton.clicked.connect(self.backend.loadMask)

        self.freeMemoryButton = QPushButton("Clear memory")
        self.freeMemoryButton.clicked.connect(self.backend.freeMemory)

        # self.startProjectionButton = QPushButton("Start projection")
        self.projectionButton = MySwitch("STOP", "red", "START", "maroon", width=50)
        self.projectionButton.clicked.connect(self.backend.handle_dmd_projection)

        self.dmdContainerLayout.addWidget(self.dmdConnectButton, 0, 0, 1, 1)
        self.dmdContainerLayout.addWidget(self.dmdIlluminateAllButton, 1, 2, 1, 1)
        self.dmdContainerLayout.addWidget(self.dmdRegistrationButton, 1, 0, 1, 1)
        self.dmdContainerLayout.addWidget(self.loadMaskButton, 0, 1, 1, 1)
        self.dmdContainerLayout.addWidget(self.freeMemoryButton, 1, 1, 1, 1)
        self.dmdContainerLayout.addWidget(self.projectionButton, 0, 2, 1, 1)
        # self.dmdContainerLayout.addWidget(self.stopProjectionButton, 1, 2, 1, 1)

        dmdContainer.setLayout(self.dmdContainerLayout)
        self.dmdWidgetLayout.addWidget(dmdContainer, 1, 2, 1, 1)

        # ----------------------  Camera Container  ---------------------------

        # ...................... Graphics scene ............................

        self.selection_view_640 = DrawingWidget(self)
        self.selection_view_640.getView().setLimits(
            xMin=0,
            xMax=2048,
            yMin=0,
            yMax=2048,
            minXRange=2048,
            minYRange=2048,
            maxXRange=2048,
            maxYRange=2048,
        )
        self.selection_view_640.ui.roiBtn.hide()
        self.selection_view_640.ui.menuBtn.hide()
        self.selection_view_640.ui.normGroup.hide()
        self.selection_view_640.ui.roiPlot.hide()

        self.roi_list_640 = []

        self.selection_view_532 = DrawingWidget(self)
        self.selection_view_532.getView().setLimits(
            xMin=0,
            xMax=2048,
            yMin=0,
            yMax=2048,
            minXRange=2048,
            minYRange=2048,
            maxXRange=2048,
            maxYRange=2048,
        )
        self.selection_view_532.ui.roiBtn.hide()
        self.selection_view_532.ui.menuBtn.hide()
        self.selection_view_532.ui.normGroup.hide()
        self.selection_view_532.ui.roiPlot.hide()

        self.roi_list_532 = []

        self.selection_view_488 = DrawingWidget(self)
        self.selection_view_488.getView().setLimits(
            xMin=0,
            xMax=2048,
            yMin=0,
            yMax=2048,
            minXRange=2048,
            minYRange=2048,
            maxXRange=2048,
            maxYRange=2048,
        )
        self.selection_view_488.ui.roiBtn.hide()
        self.selection_view_488.ui.menuBtn.hide()
        self.selection_view_488.ui.normGroup.hide()
        self.selection_view_488.ui.roiPlot.hide()

        self.roi_list_488 = []

        self.selection_view_2p = DrawingWidget(self)
        self.selection_view_2p.getView().setLimits(
            xMin=0,
            xMax=2048,
            yMin=0,
            yMax=2048,
            minXRange=2048,
            minYRange=2048,
            maxXRange=2048,
            maxYRange=2048,
        )
        self.selection_view_2p.ui.roiBtn.hide()
        self.selection_view_2p.ui.menuBtn.hide()
        self.selection_view_2p.ui.normGroup.hide()
        self.selection_view_2p.ui.roiPlot.hide()

        self.roi_list_2p = []

        self.movieProjectionView = SquareImageView()
        self.movieProjectionView.getView().setLimits(
            xMin=0,
            xMax=2048,
            yMin=0,
            yMax=2048,
            minXRange=2048,
            minYRange=2048,
            maxXRange=2048,
            maxYRange=2048,
        )
        self.movieProjectionView.ui.roiBtn.hide()
        self.movieProjectionView.ui.menuBtn.hide()
        self.movieProjectionView.ui.normGroup.hide()
        self.movieProjectionView.ui.roiPlot.hide()

        self.stacked_selection_views = QtWidgets.QStackedWidget()
        self.stacked_selection_views.addWidget(self.selection_view_640)
        self.stacked_selection_views.addWidget(self.selection_view_532)
        self.stacked_selection_views.addWidget(self.selection_view_488)
        self.stacked_selection_views.addWidget(self.selection_view_2p)

        self.stackedImageViews = QtWidgets.QStackedWidget()
        self.stackedImageViews.addWidget(self.movieProjectionView)
        self.stackedImageViews.addWidget(self.stacked_selection_views)

        # ..................................................................

        self.dmdWidgetLayout.addWidget(self.stackedImageViews, 1, 0, 2, 2)

        # ----------------------  DMD mask generation Container  --------------

        self.maskGeneratorContainer = roundQGroupBox("Mask generator")
        self.maskGeneratorContainer.setMaximumHeight(500)
        self.maskGeneratorContainerLayout = QGridLayout()

        self.maskGeneratorLayout = QGridLayout()
        self.maskGeneratorContainer.setLayout(self.maskGeneratorLayout)

        self.maskModesWidget = QWidget()
        self.maskModesWidget.setMaximumHeight(50)
        self.maskModesLayout = QHBoxLayout()
        self.maskModesWidget.setLayout(self.maskModesLayout)
        self.maskMovie = QPushButton("Movie")
        self.maskLaser488 = QPushButton("488")
        self.maskLaser532 = QPushButton("532")
        self.maskLaser640 = QPushButton("640")
        self.maskLaser2p = QPushButton("2P")
        self.maskMovie.clicked.connect(self.switch_to_movie_projection)
        self.maskLaser488.clicked.connect(lambda: self.switch_drawing_laser("488"))
        self.maskLaser532.clicked.connect(lambda: self.switch_drawing_laser("532"))
        self.maskLaser640.clicked.connect(lambda: self.switch_drawing_laser("640"))
        self.maskLaser2p.clicked.connect(lambda: self.switch_drawing_laser("2p"))
        self.maskModesLayout.addWidget(self.maskMovie)
        self.maskModesLayout.addWidget(self.maskLaser488)
        self.maskModesLayout.addWidget(self.maskLaser532)
        self.maskModesLayout.addWidget(self.maskLaser640)
        self.maskModesLayout.addWidget(self.maskLaser2p)

        self.maskGeneratorLayout.addWidget(self.maskModesWidget, 0, 0, 1, 4)

        self.maskGeneratorStackLayer0 = QWidget()
        self.maskGeneratorStackLayer1 = QWidget()
        self.maskGeneratorStackLayer2 = QWidget()
        self.maskGeneratorStack = QStackedWidget()
        self.maskGeneratorLayout.addWidget(self.maskGeneratorStack, 1, 0)
        self.maskGeneratorStack.addWidget(self.maskGeneratorStackLayer0)
        self.maskGeneratorStack.addWidget(self.maskGeneratorStackLayer1)
        self.maskGeneratorStack.addWidget(self.maskGeneratorStackLayer2)
        self.maskGeneratorStack.setCurrentIndex(1)

        self.maskGeneratorStackContainer = QGroupBox()
        self.maskGeneratorStackContainer.setMaximumHeight(400)
        # self.maskGeneratorStackContainer.setStyleSheet("QGroupBox{padding-top:15px; margin-top:-15px}")
        self.maskGeneratorStackContainerLayout = QGridLayout()

        self.maskGeneratorStackContainer.setLayout(
            self.maskGeneratorStackContainerLayout
        )
        self.maskGeneratorStackContainerLayout.addWidget(self.maskGeneratorStack)
        self.maskGeneratorLayout.addWidget(self.maskGeneratorStackContainer, 1, 0, 1, 4)

        self.maskGeneratorStackLayer0Layout = QGridLayout()
        self.maskGeneratorStackLayer0.setLayout(self.maskGeneratorStackLayer0Layout)
        self.maskGeneratorStackLayer1Layout = QGridLayout()
        self.maskGeneratorStackLayer1.setLayout(self.maskGeneratorStackLayer1Layout)
        self.maskGeneratorStackLayer2Layout = QGridLayout()
        self.maskGeneratorStackLayer2.setLayout(self.maskGeneratorStackLayer2Layout)

        # Three buttons for setting selection mode
        self.selectionModes = ["Polygon", "Freehand", "From file"]
        self.selectionModeMenu = QComboBox()
        self.selectionModeMenu.addItems(self.selectionModes)
        self.selectionModeMenu.currentIndexChanged.connect(self.selection_mode_changed)

        self.snapWhiteImageButton = QPushButton("Snap FOV")
        self.snapWhiteImageButton.clicked.connect(self.backend.snap_fov)
        self.maskGeneratorStackLayer1Layout.addWidget(
            self.snapWhiteImageButton, 0, 1, 1, 1
        )

        label = QLabel("Selection mode:")
        label.setAlignment(Qt.AlignRight)
        self.maskGeneratorStackLayer1Layout.addWidget(label, 1, 0, 1, 1)
        self.maskGeneratorStackLayer1Layout.addWidget(
            self.selectionModeMenu, 1, 1, 1, 1
        )

        # Movie projection
        self.movieFromFileContainer = QWidget()
        self.movieFromFileLayout = QGridLayout()
        self.loadFolderNameTextBox = QtWidgets.QLineEdit(self)
        self.loadFolderNameTextBox.setPlaceholderText("path to folder..")
        self.setFrameRateTextBox = QtWidgets.QLineEdit(self)
        self.setFrameRateTextBox.setValidator(QtGui.QIntValidator())
        self.browseFolderButton = QPushButton("Browse")
        self.loadFolderButton = QPushButton("Load movie")
        self.invertMovieButton = QCheckBox("Invert")
        self.movieRepeatSwitch = MySwitch("ON", "red", "OFF", "maroon", width=32)

        self.browseFolderButton.clicked.connect(self.browse_folders)
        self.loadFolderButton.clicked.connect(self.backend.loadFolder)
        self.movieFromFileLayout.addWidget(self.loadFolderNameTextBox, 0, 0, 1, 1)
        self.movieFromFileLayout.addWidget(self.browseFolderButton, 0, 1, 1, 1)
        self.movieFromFileLayout.addWidget(self.invertMovieButton, 1, 0, 1, 1)
        self.movieFromFileLayout.addWidget(self.loadFolderButton, 1, 1, 1, 1)
        self.movieFromFileLayout.addWidget(QLabel(u"Period (µs):"), 2, 0, 1, 1)
        self.movieFromFileLayout.addWidget(self.setFrameRateTextBox, 2, 1, 1, 1)
        self.movieFromFileLayout.addWidget(QLabel("Repeat:"), 3, 0, 1, 1)
        self.movieFromFileLayout.addWidget(self.movieRepeatSwitch, 3, 1, 1, 1)
        self.movieFromFileContainer.setLayout(self.movieFromFileLayout)
        self.maskGeneratorStackLayer0Layout.addWidget(self.movieFromFileContainer)

        # 2P
        self.twoPSelectionControlContainer = QGroupBox()
        # self.twoPSelectionControlContainer.setStyleSheet("QGroupBox{padding-top:15px; margin-top:-15px}")
        self.twoPSelectionControlLayout = QGridLayout()
        self.twoPSelectionControlContainer.setLayout(self.twoPSelectionControlLayout)
        self.twoPAddRectangleRoi = QPushButton("Add ROI")
        self.twoPAddRectangleRoi.clicked.connect(self.add_polygon_roi_for_2p)
        self.maskGeneratorStackLayer2Layout.addWidget(self.twoPAddRectangleRoi, 0, 0)

        # Polygon selection
        self.polygonSelectionControlContainer = QGroupBox()
        # self.polygonSelectionControlContainer.setStyleSheet("QGroupBox{padding-top:15px; margin-top:-15px}")
        self.polygonSelectionControlLayout = QVBoxLayout()
        self.addRoiButton = QPushButton("Add ROI")
        self.createMaskButton = QPushButton("Create mask")
        self.removeSelectionButton = QPushButton("Clear selection")
        self.previewMaskButton = QPushButton("Preview mask")
        self.addRoiButton.clicked.connect(self.add_polygon_roi)
        self.createMaskButton.clicked.connect(self.backend.createMask)
        self.removeSelectionButton.clicked.connect(self.remove_selection)
        self.previewMaskButton.clicked.connect(self.popup_window_preview_mask)
        self.polygonSelectionControlLayout.addWidget(self.addRoiButton)
        self.polygonSelectionControlLayout.addWidget(self.createMaskButton)
        self.polygonSelectionControlLayout.addWidget(self.removeSelectionButton)
        self.polygonSelectionControlLayout.addWidget(self.previewMaskButton)
        self.polygonSelectionControlContainer.setLayout(
            self.polygonSelectionControlLayout
        )
        self.polygonSelectionOptionsContainer = QGroupBox()
        # self.polygonSelectionOptionsContainer.setStyleSheet("QGroupBox{padding-top:15px; margin-top:-15px}")
        self.polygonSelectionOptionsLayout = QVBoxLayout()
        self.polygonSaveMaskButton = QCheckBox("Save mask")
        self.polygonFillContourButton = QCheckBox("Fill contour")
        self.polygonInvertMaskButton = QCheckBox("Invert mask")
        self.polygonSelectionOptionsLayout.addWidget(self.polygonSaveMaskButton)
        self.polygonSelectionOptionsLayout.addWidget(self.polygonFillContourButton)
        self.polygonSelectionOptionsLayout.addWidget(self.polygonInvertMaskButton)
        self.polygonSelectionOptionsContainer.setLayout(
            self.polygonSelectionOptionsLayout
        )

        self.polygonSelectionThicknessContainer = QGroupBox()
        # self.polygonSelectionThicknessContainer.setStyleSheet("QGroupBox{padding-top:15px; margin-top:-15px}")
        self.polygonThicknessSpinBox = QSpinBox()
        self.polygonThicknessSpinBox.setRange(1, 10)
        self.polygonThicknessLayout = QHBoxLayout()
        self.polygonSelectionThicknessContainer.setLayout(self.polygonThicknessLayout)
        self.polygonThicknessLayout.addWidget(QLabel("Line thickness:"))
        self.polygonThicknessLayout.addWidget(self.polygonThicknessSpinBox)

        self.polygonSelectionContainer = QWidget()
        self.polygonSelectionLayout = QGridLayout()
        self.polygonSelectionContainer.setLayout(self.polygonSelectionLayout)
        self.polygonSelectionLayout.addWidget(
            self.polygonSelectionControlContainer, 0, 0, 2, 1
        )
        self.polygonSelectionLayout.addWidget(
            self.polygonSelectionOptionsContainer, 0, 1
        )
        self.polygonSelectionLayout.addWidget(
            self.polygonSelectionThicknessContainer, 1, 1
        )

        # Freehand selection
        self.freehandSelectionControlContainer = QGroupBox()
        # self.freehandSelectionControlContainer.setStyleSheet("QGroupBox{padding-top:15px; margin-top:-20px}")
        self.freehandSelectionControlLayout = QVBoxLayout()
        self.createMaskButton = QPushButton("Create mask")
        self.removeSelectionButton = QPushButton("Clear selection")
        self.previewMaskButton = QPushButton("Preview mask")
        self.createMaskButton.clicked.connect(self.backend.createMask)
        self.removeSelectionButton.clicked.connect(self.remove_selection)
        self.previewMaskButton.clicked.connect(self.popup_window_preview_mask)
        self.freehandSelectionControlLayout.addWidget(self.createMaskButton)
        self.freehandSelectionControlLayout.addWidget(self.removeSelectionButton)
        self.freehandSelectionControlLayout.addWidget(self.previewMaskButton)
        self.freehandSelectionControlContainer.setLayout(
            self.freehandSelectionControlLayout
        )
        self.freehandSelectionOptionsContainer = QGroupBox()
        # self.freehandSelectionOptionsContainer.setStyleSheet("QGroupBox{padding-top:15px; margin-top:-15px}")
        self.freehandSelectionOptionsLayout = QVBoxLayout()
        self.freehandSaveMaskButton = QCheckBox("Save mask")
        self.freehandFillContourButton = QCheckBox("Fill contour")
        self.freehandInvertMaskButton = QCheckBox("Invert mask")
        self.freehandSelectionOptionsLayout.addWidget(self.freehandSaveMaskButton)
        self.freehandSelectionOptionsLayout.addWidget(self.freehandFillContourButton)
        self.freehandSelectionOptionsLayout.addWidget(self.freehandInvertMaskButton)
        self.freehandSelectionOptionsContainer.setLayout(
            self.freehandSelectionOptionsLayout
        )
        self.freehandSelectionThicknessContainer = QGroupBox()
        # self.freehandSelectionThicknessContainer.setStyleSheet("QGroupBox{padding-top:15px; margin-top:-15px}")
        self.freehandThicknessSpinBox = QSpinBox()
        self.freehandThicknessSpinBox.setRange(1, 10)
        self.freehandThicknessLayout = QHBoxLayout()
        self.freehandSelectionThicknessContainer.setLayout(self.freehandThicknessLayout)
        self.freehandThicknessLayout.addWidget(QLabel("Line thickness:"))
        self.freehandThicknessLayout.addWidget(self.freehandThicknessSpinBox)

        self.freehandSelectionContainer = QWidget()
        self.freehandSelectionLayout = QGridLayout()
        self.freehandSelectionLayout.addWidget(
            self.freehandSelectionControlContainer, 0, 0, 2, 1
        )
        self.freehandSelectionLayout.addWidget(
            self.freehandSelectionOptionsContainer, 0, 1
        )
        self.freehandSelectionLayout.addWidget(
            self.freehandSelectionThicknessContainer, 1, 1
        )
        self.freehandSelectionContainer.setLayout(self.freehandSelectionLayout)

        # Load from file
        self.imageFromFileContainer = QWidget()
        self.imageFromFileLayout = QGridLayout()
        self.loadFileNameTextBox = QtWidgets.QLineEdit(self)
        self.loadFileNameTextBox.setPlaceholderText("path to file..")
        self.browseFileButton = QPushButton("Browse")
        self.previewMaskButton = QPushButton("Preview mask")
        self.browseFileButton.clicked.connect(self.browse_files)
        self.previewMaskButton.clicked.connect(self.popup_window_preview_mask)
        self.imageFromFileLayout.addWidget(self.loadFileNameTextBox, 0, 0, 1, 2)
        self.imageFromFileLayout.addWidget(self.browseFileButton, 0, 1, 1, 1)
        self.imageFromFileLayout.addWidget(self.previewMaskButton, 1, 0, 1, 1)
        self.imageFromFileContainer.setLayout(self.imageFromFileLayout)

        # Stacked Layout containing control for selection modes
        self.stackedSelectionOptionsContainer = QtWidgets.QStackedWidget()
        self.stackedSelectionOptionsContainer.addWidget(self.polygonSelectionContainer)
        self.stackedSelectionOptionsContainer.addWidget(self.freehandSelectionContainer)
        self.stackedSelectionOptionsContainer.addWidget(self.imageFromFileContainer)
        self.stackedSelectionOptionsContainer.setCurrentIndex(0)
        self.maskGeneratorStackLayer1Layout.addWidget(
            self.stackedSelectionOptionsContainer, 2, 0, 1, 2
        )

        self.dmdWidgetLayout.addWidget(self.maskGeneratorContainer, 2, 2, 1, 1)

        self.switch_drawing_laser("640")
        self.selectionModeMenu.setCurrentIndex(1)
        self.selectionModeMenu.setCurrentIndex(0)
        self.update_buttons()

        self.backend.read_transformations_from_file()
        try:
            self.backend.connectDMD()
        except:
            self.dmdConnectButton.setChecked(False)
        else:
            self.dmdConnectButton.setChecked(True)

    def ImageViewResizeEvent(self, event):
        # Create a square base size of 10x10 and scale it to the new size
        # maintaining aspect ratio.
        new_size = QtCore.QSize(10, 10)
        new_size.scale(event.size(), QtCore.Qt.KeepAspectRatio)
        self.stackedImageViews.resize(new_size)

    def select_lasers_popup(self):
        self.popup_window = QWidget()
        self.popup_window.setGeometry(300, 300, 50, 100)
        popup_window_layout = QGridLayout()
        self.popup_window.setLayout(popup_window_layout)
        label = QLabel("640")
        label.setAlignment(Qt.AlignRight)
        popup_window_layout.addWidget(label, 0, 0)
        label = QLabel("532")
        label.setAlignment(Qt.AlignRight)
        popup_window_layout.addWidget(label, 1, 0)
        label = QLabel("488")
        label.setAlignment(Qt.AlignRight)
        popup_window_layout.addWidget(label, 2, 0)

        self.checkbox_640 = QCheckBox()
        self.checkbox_532 = QCheckBox()
        self.checkbox_488 = QCheckBox()
        popup_window_layout.addWidget(self.checkbox_640, 0, 1)
        popup_window_layout.addWidget(self.checkbox_532, 1, 1)
        popup_window_layout.addWidget(self.checkbox_488, 2, 1)
        close_button = QPushButton("Start registration")
        close_button.clicked.connect(self.select_lasers_popup_close)
        popup_window_layout.addWidget(close_button, 3, 0, 1, 2)
        self.popup_window.show()

    def select_lasers_popup_close(self, e):
        lasers_to_register = []
        if self.checkbox_488.isChecked():
            lasers_to_register.append("488")
        if self.checkbox_532.isChecked():
            lasers_to_register.append("532")
        if self.checkbox_640.isChecked():
            lasers_to_register.append("640")

        if not lasers_to_register:
            return

        self.popup_window.close()
        self.backend.dmd_register(lasers_to_register)

    def popup_window_preview_mask(self, e):
        self.popup_window = QWidget()
        self.popup_window.setGeometry(300, 300, 500, 500)
        popup_window_layout = QGridLayout()
        self.popup_window.setLayout(popup_window_layout)

        mask_preview_popup_window = SquareImageView()
        mask_preview_popup_window.ui.roiBtn.hide()
        mask_preview_popup_window.ui.menuBtn.hide()
        mask_preview_popup_window.ui.normGroup.hide()
        mask_preview_popup_window.ui.roiPlot.hide()
        mask_preview_popup_window.ui.histogram.hide()
        mask_preview_popup_window.setImage(
            self.backend.mask[self.drawing_laser].transpose(), levels=[0, 1]
        )

        popup_window_layout.addWidget(mask_preview_popup_window)
        self.popup_window.show()

    def set_camera_image(self, sig):
        self.backend.image = sig
        self.update_image()

    def switch_drawing_laser(self, laser):
        self.maskGeneratorStack.setCurrentIndex(1)
        self.maskMovie.setStyleSheet("background-color: light gray")

        self.drawing_laser = laser
        if laser == "488":
            self.maskLaser488.setStyleSheet("background-color: teal")
            self.maskLaser532.setStyleSheet("background-color: light gray")
            self.maskLaser640.setStyleSheet("background-color: light gray")
            self.maskLaser2p.setStyleSheet("background-color: light gray")
            self.stacked_selection_views.setCurrentIndex(2)
            self.maskGeneratorStack.setCurrentIndex(1)
        elif laser == "532":
            self.maskLaser488.setStyleSheet("background-color: light gray")
            self.maskLaser532.setStyleSheet("background-color: darkgreen")
            self.maskLaser640.setStyleSheet("background-color: light gray")
            self.maskLaser2p.setStyleSheet("background-color: light gray")
            self.stacked_selection_views.setCurrentIndex(1)
            self.maskGeneratorStack.setCurrentIndex(1)
        elif laser == "640":
            self.maskLaser488.setStyleSheet("background-color: light gray")
            self.maskLaser532.setStyleSheet("background-color: light gray")
            self.maskLaser640.setStyleSheet("background-color: maroon")
            self.maskLaser2p.setStyleSheet("background-color: light gray")
            self.stacked_selection_views.setCurrentIndex(0)
            self.maskGeneratorStack.setCurrentIndex(1)
        else:
            self.maskLaser488.setStyleSheet("background-color: light gray")
            self.maskLaser532.setStyleSheet("background-color: light gray")
            self.maskLaser640.setStyleSheet("background-color: light gray")
            self.maskLaser2p.setStyleSheet("background-color: yellow")
            self.stacked_selection_views.setCurrentIndex(3)
            self.maskGeneratorStack.setCurrentIndex(2)

    def add_polygon_roi(self):
        if self.drawing_laser == "640":
            view = self.selection_view_640
            roi_list = self.roi_list_640
        elif self.drawing_laser == "532":
            view = self.selection_view_532
            roi_list = self.roi_list_532
        else:
            view = self.selection_view_488
            roi_list = self.roi_list_488

        x = (view.getView().viewRect().x()) * 0.3
        y = (view.getView().viewRect().y()) * 0.3
        a = (view.getView().viewRect().width() + x) * 0.3
        b = (view.getView().viewRect().height() + y) * 0.3
        c = (view.getView().viewRect().width() + x) * 0.7
        d = (view.getView().viewRect().height() + y) * 0.7
        polygon_roi = pg.PolyLineROI(
            [[a, b], [c, b], [c, d], [a, d]],
            pen=view.pen,
            closed=True,
            movable=True,
            removable=True,
        )

        roi_list.append(polygon_roi)

        view.getView().addItem(polygon_roi)

    def add_polygon_roi_for_2p(self):
        view = self.selection_view_2p

        x0 = (view.getView().viewRect().width()) * 0.5
        y0 = (view.getView().viewRect().height()) * 0.5

        delta_angle = np.pi / 4
        radius_x = view.getView().viewRect().width() * 0.3
        radius_y = view.getView().viewRect().height() * 0.3

        handles = []
        for i in range(8):
            angle = delta_angle * i
            handles.append(
                [
                    x0 + radius_x * 0.8 * np.sin(angle),
                    y0 + radius_y * 0.8 * np.cos(angle),
                ]
            )

        rectangle_roi = pg.PolyLineROI(
            handles, pen=view.pen, closed=True, movable=True, removable=True
        )
        view.getView().addItem(rectangle_roi)
        self.roi_list_2p.append(rectangle_roi)

    def add_freehand_roi(self, roi):
        if self.drawing_laser == "640":
            view = self.selection_view_640
            roi_list = self.roi_list_640
        elif self.drawing_laser == "532":
            view = self.selection_view_532
            roi_list = self.roi_list_532
        else:
            view = self.selection_view_488
            roi_list = self.roi_list_488

        roi_list.append(roi)

    def normalOutputWritten(self, text):
        if __name__ == "__main__":
            """Append text to the QTextEdit."""
            cursor = self.noticeBoard.textCursor()
            cursor.movePosition(QtGui.QTextCursor.End)
            cursor.insertText(text + "\n")
            self.noticeBoard.setTextCursor(cursor)
            self.noticeBoard.ensureCursorVisible()
        else:
            self.sig_console_print.emit(text)

    def selection_mode_changed(self):
        self.selectionModeIndex = self.selectionModeMenu.currentIndex()
        self.stackedImageViews.setCurrentIndex(1)
        if self.selectionModeIndex == 0:
            self.set_polygon_selection_mode()
        elif self.selectionModeIndex == 1:
            self.set_freehand_selection_mode()
        else:
            self.set_load_image_mode()

    def switch_to_movie_projection(self):
        self.maskGeneratorStack.setCurrentIndex(0)
        self.maskMovie.setStyleSheet("background-color: gray")
        self.maskLaser488.setStyleSheet("background-color: light gray")
        self.maskLaser532.setStyleSheet("background-color: light gray")
        self.maskLaser640.setStyleSheet("background-color: light gray")
        self.maskLaser2p.setStyleSheet("background-color: light gray")
        self.stackedImageViews.setCurrentIndex(0)

    def set_polygon_selection_mode(self):
        """
        This function changes the settings to enable polygon ROI drawing.
        """

        self.selectionMode = "polygonMode"
        self.stackedSelectionOptionsContainer.setCurrentIndex(0)

        self.selection_view_640.enable_drawing(False)
        self.selection_view_532.enable_drawing(False)
        self.selection_view_488.enable_drawing(False)

    def set_freehand_selection_mode(self):
        """
        This function changes the settings to enable freehand ROI drawing.
        """

        self.selectionMode = "freehandMode"
        self.stackedSelectionOptionsContainer.setCurrentIndex(1)

        self.selection_view_640.enable_drawing(True)
        self.selection_view_532.enable_drawing(True)
        self.selection_view_488.enable_drawing(True)

    def set_load_image_mode(self):
        """
        This function changes the settings to enable loading ROI from file.
        """
        self.selectionMode = "loadImage"
        self.stackedSelectionOptionsContainer.setCurrentIndex(2)

        self.selection_view_640.enable_drawing(False)
        self.selection_view_532.enable_drawing(False)
        self.selection_view_488.enable_drawing(False)

    def browse_files(self):
        """
        Open a file manager to browse through files. Save selected file's path.
        """
        self.backend.loadFileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select file", "./CoordinateManager/Images/", "(*.png)"
        )
        self.loadFileNameTextBox.setText(self.backend.loadFileName)
        self.backend.loadFile()

    def update_image(self):
        self.selection_view_640.setImage(self.backend.image)
        self.selection_view_640.getView().setLimits(minXRange=0, minYRange=0)

        self.selection_view_532.setImage(self.backend.image)
        self.selection_view_532.getView().setLimits(minXRange=0, minYRange=0)

        self.selection_view_488.setImage(self.backend.image)
        self.selection_view_488.getView().setLimits(minXRange=0, minYRange=0)

        self.movieProjectionView.setImage(self.backend.image)
        self.movieProjectionView.getView().setLimits(minXRange=0, minYRange=0)

        self.selection_view_2p.setImage(self.backend.image)
        self.selection_view_2p.getView().setLimits(minXRange=0, minYRange=0)

    # def update_mask_preview(self):
    #     if self.drawing_laser == '488':
    #         self.maskPreviewImageItem488.setImage(self.backend.mask[self.drawing_laser].transpose(), levels=[0,1])
    #     elif self.drawing_laser == '532':
    #         self.maskPreviewImageItem532.setImage(self.backend.mask[self.drawing_laser].transpose(), levels=[0,1])
    #     else:
    #         self.maskPreviewImageItem640.setImage(self.backend.mask[self.drawing_laser].transpose(), levels=[0,1])

    def browse_folders(self):
        """
        Open a file manager to browse through files. Save selected folder's path.
        """
        self.backend.loadFolderName = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder", "./CoordinateManager/Images/"
        )
        self.loadFolderNameTextBox.setText(self.backend.loadFolderName)

    def get_drawing_laser_view(self):
        if self.drawing_laser == "640":
            return self.selection_view_640
        elif self.drawing_laser == "532":
            return self.selection_view_532
        else:
            return self.selection_view_488

    def get_drawing_laser_roi_list(self):
        if self.drawing_laser == "640":
            return self.roi_list_640
        elif self.drawing_laser == "532":
            return self.roi_list_532
        else:
            return self.roi_list_488

    def empty_roi_list(self):
        if self.drawing_laser == "640":
            self.roi_list_640 = []
        elif self.drawing_laser == "532":
            self.roi_list_532 = []
        else:
            self.roi_list_488 = []

    def remove_selection(self):
        """
        This function removes the freehand ROI that is drawn from the screen.
        """
        view = self.get_drawing_laser_view()
        roi_list = self.get_drawing_laser_roi_list()
        for roi in roi_list:
            view.removeItem(roi)

        self.empty_roi_list()
        self.createMask()

    def lasers_status_changed(self, lasers_status):
        self.backend.lasers_status = lasers_status

    def update_buttons(self):
        pass
        # # Check whether registration procedure is running
        # if self.backend.flag_registrating:
        #     self.dmdRegistrationButton.setEnabled(False)
        #     self.backend.flag_projecting = True
        # else:
        #     self.dmdRegistrationButton.setEnabled(True)

        # # Check whether DMD is connected and how its status is
        # if not self.backend.flag_dmd_connected:
        #     self.snapWhiteImageButton.setEnabled(False)
        #     self.loadMaskButton.setEnabled(False)
        #     self.freeMemoryButton.setEnabled(False)
        #     self.dmdRegistrationButton.setEnabled(False)

        # elif not self.backend.flag_mask_in_memory:
        #     self.snapWhiteImageButton.setEnabled(True)
        #     self.freeMemoryButton.setEnabled(False)
        #     self.dmdRegistrationButton.setEnabled(True)

        #     if self.backend.flag_mask_created:
        #         self.loadMaskButton.setEnabled(True)
        #     else:
        #         self.loadMaskButton.setEnabled(False)

        # elif not self.backend.flag_projecting:
        #     self.snapWhiteImageButton.setEnabled(True)
        #     self.loadMaskButton.setEnabled(False)
        #     self.freeMemoryButton.setEnabled(True)
        #     self.dmdRegistrationButton.setEnabled(True)

        # elif self.backend.flag_projecting:
        #     self.snapWhiteImageButton.setEnabled(False)
        #     self.loadMaskButton.setEnabled(False)
        #     self.freeMemoryButton.setEnabled(False)
        #     self.dmdRegistrationButton.setEnabled(False)


class DMDWidget:
    def __init__(self, parent):
        """

        Initiate class by instantiating DMD from ALP4 software and initializing
        with device serial number as argument.

        """
        self.ui_widget = parent

        # Set flags to default
        self.flag_dmd_connected = False
        self.flag_registrating = False
        self.flag_registered = False
        self.flag_mask_created = False
        self.flag_mask_in_memory = False
        self.flag_projecting = False
        self.flag_camera_connected = False

        self.mask = {}
        self.mask_transformed = {}
        self.dict_transformations = {}

        self.lasers = ["488", "532", "640"]
        for laser in self.lasers:
            self.mask[laser] = np.zeros((1024, 768))
            self.mask_transformed[laser] = np.zeros((1024, 768))

        self.lasers_status = {}
        self.lasers_status["488"] = [False, 0]
        self.lasers_status["532"] = [False, 0]
        self.lasers_status["640"] = [False, 0]

    def read_transformations_from_file(self):
        try:
            with open(
                "CoordinateManager/Registration/transformation.txt", "r"
            ) as json_file:
                self.dict_transformations = json.load(json_file)
        except:
            self.ui_widget.normalOutputWritten(
                "No transformation could be loaded from previous registration run."
            )
            return

    def handle_dmd_connection(self):
        if self.ui_widget.dmdConnectButton.isChecked():
            self.connectDMD()
        else:
            self.disconnectDMD()

    def connectDMD(self):
        """
        Check whether there is an open DMD connection. If not, create one.
        """
        pass
        if self.flag_dmd_connected:
            self.ui_widget.normalOutputWritten("DMD connected")
            return

        # Load the Vialux .dll
        cdir = os.getcwd() + "\\CoordinateManager"
        self.DMD = ALP4(
            version="4.3", libDir=r"" + cdir
        )  # Use version 4.3 for the alp4395.dll

        # Initialize the device
        self.DMD.Initialize(13388)

        self.ui_widget.normalOutputWritten("DMD connected")
        self.flag_dmd_connected = True
        self.ui_widget.update_buttons()

    def disconnectDMD(self):
        """

        Check whether there is an open DMD connection. If so, close connection

        """

        if not self.flag_dmd_connected:
            return

        # Clear onboard memory and disconnect
        self.deallocDevide()

        self.DMD = None
        self.ui_widget.normalOutputWritten("DMD disconnected")

        self.flag_dmd_connected = False
        self.ui_widget.update_buttons()

    #     def loadMask(self, mask = None):
    #         """

    #         Load image or image sequence to onboard memory of DMD.
    #         In case of binary illumination, bit depth of image should be 1.

    #         param mask: 2d binary numpy array
    #         type mask: Illumination mask

    #         """

    #         if isinstance(mask, np.ndarray):
    #             img_seq = mask
    #         else:
    #             img_seq = self.mask_transformed[self.ui_widget.drawing_laser]

    #         if len(img_seq.shape) == 2:
    #             self.seq_length = 1
    #             self.image = img_seq.ravel()
    #         else:
    #             self.seq_length = img_seq.shape[2]

    #             self.image = np.concatenate([img_seq[:,:,0].ravel(), img_seq[:,:,1].ravel()])
    #             for i in range(2,self.seq_length):
    #                 self.image = np.hstack([self.image, img_seq[:,:,i].ravel()])

    # #            self.image = np.squeeze(np.reshape(img_seq, (1, -1), order='F'))

    #         self.image = (self.image > 0)*1 #First part makes it True/False, multiplying by 1 converts it to binary

    #         # Binary amplitude image (0 or 1)
    #         bitDepth = 1
    #         self.image*=(2**8-1)
    #         self.image = self.image.astype(int)
    #         # Allocate the onboard memory for the image sequence
    #         # nbImg defines the number of masks
    #         DMD = DMDActuator.DMDActuator()
    #         DMD.send_data_to_DMD(self.image)
    #         DMD.deallocDevice()
    #         # self.DMD.SeqAlloc(nbImg = self.seq_length, bitDepth = bitDepth)

    #         # Send the image sequence as a 1D list/array/numpy array
    #         # self.DMD.SeqPut(imgData = self.image)

    #         self.flag_mask_in_memory = True
    #         self.ui_widget.update_buttons()

    def loadMask(self, mask=None):
        if isinstance(mask, np.ndarray):
            img_seq = mask
        else:
            img_seq = self.mask_transformed[self.ui_widget.drawing_laser]

        DMD = DMDActuator.DMDActuator()
        DMD.send_data_to_DMD(img_seq)
        DMD.disconnect_DMD()

        self.flag_mask_in_memory = True
        self.ui_widget.update_buttons()

    def startProjection(self):
        frame_rate = self.ui_widget.setFrameRateTextBox.text()
        if frame_rate == "":
            frame_rate = None

        DMD = DMDActuator.DMDActuator()

        if frame_rate != None:
            DMD.set_timing(illuminationTime=frame_rate)
            loop = self.ui_widget.movieRepeatSwitch.isChecked()
            DMD.start_projection(loop=loop)
        else:
            DMD.start_projection()

        DMD.disconnect_DMD()

        self.flag_projecting = True
        self.ui_widget.update_buttons()

    def handle_dmd_projection(self):
        if self.ui_widget.projectionButton.isChecked():
            self.startProjection()
        else:
            self.stopProjection()

    # def startProjection(self, frame_rate = None):
    #     """

    #     Illuminate the sample using a light pattern that is defined by the mask.

    #     """
    #     ### Code needed to display a sequence with waiting time frame_rate.

    #     # Run the sequence in an infinite loop
    #     frame_rate = self.ui_widget.setFrameRateTextBox.text()
    #     if frame_rate == '' and self.seq_length > 1:
    #         self.ui_widget.normalOutputWritten("Please enter frame rate in microseconds")
    #         return
    #     elif not frame_rate == '':
    #         frame_rate = int(frame_rate)
    #     else:
    #         frame_rate = None

    #     if frame_rate != None:
    #         self.DMD.SetTiming(illuminationTime=frame_rate)
    #         loop = self.ui_widget.movieRepeatSwitch.isChecked()
    #         self.DMD.Run(loop = loop)
    #     else:
    #         self.DMD.Run()

    #     self.flag_projecting = True
    #     self.ui_widget.update_buttons()

    def stopProjection(self):
        """
        Stop illuminating the ROI. Image stays in memory of DMD
        """
        # Stop the sequence display
        self.DMD.Halt()

        self.flag_projecting = False
        self.ui_widget.update_buttons()

    def freeMemory(self):
        """

        Free the onboard RAM of the DMD and disconnect the DMD.

        """
        # Free the sequence from the onboard memory
        self.DMD.Halt()

        # If there has no seq be loaded to memory, this function gives error.
        # Therefore, use try:
        try:
            self.DMD.FreeSeq()
        except:
            pass

        self.flag_mask_in_memory = False
        self.ui_widget.update_buttons()

    def deallocDevide(self):
        # De-allocate the device
        self.freeMemory()
        self.DMD.Free()

    def dmd_illuminate_all(self):
        print()
        self.loadMask(mask=np.ones((768, 1024)))
        self.ui_widget.projectionButton.setChecked(True)
        self.startProjection()

    def snap_fov(self):
        self.loadMask(mask=np.ones((768, 1024)))
        self.startProjection()

        try:
            cam = CamActuator()
            cam.initializeCamera()
        except:
            print(sys.exc_info())
            self.ui_widget.normalOutputWritten("Unable to connect Hamamatsu camera")
            return

        cam.setROI(0, 0, 2048, 2048)
        self.image = cam.SnapImage(0.04)
        cam.Exit()

        self.stopProjection()
        self.freeMemory()

        self.ui_widget.update_image()
        self.ui_widget.update_buttons()

    def save_mask(self):
        if not (
            self.ui_widget.freehandSaveMaskButton.isChecked()
            and self.ui_widget.selectionMode == "freehandMode"
            or self.ui_widget.polygonSaveMaskButton.isChecked()
            and self.ui_widget.selectionMode == "polygonMode"
        ):
            return

        image = Image.fromarray(
            (self.mask[self.ui_widget.drawing_laser] * 255).astype(np.uint8)
        )
        image = image.convert("L")
        date_time = datetime.datetime.now().timetuple()
        image_id = ""
        for i in range(5):
            image_id += str(date_time[i]) + "_"
        image_id += str(date_time[5]) + "_" + self.ui_widget.drawing_laser
        image.save("CoordinateManager/Saved_masks/" + image_id + ".png", "PNG")

    def galvos_scan_mask(self):
        reference_length = 100
        tp_digital = np.dtype(
            [("Waveform", bool, (reference_length,)), ("Sepcification", "U20")]
        )

        if len(self.ui_widget.roi_list_2p) == 0:
            self.ui_widget.normalOutputWritten(
                "No region selected for 2p galvo scanning"
            )
            return

        roi_handle_positions = self.ui_widget.roi_list_2p[0].getLocalHandlePositions()

        for idx, pos in enumerate(roi_handle_positions):
            roi_handle_positions[
                idx
            ] = self.ui_widget.selection_view_2p.getView().mapToView(pos[1])

        num_vertices = len(roi_handle_positions)
        vertices = np.zeros([num_vertices, 2])

        for idx, vertex in enumerate(roi_handle_positions):
            vertices[idx, :] = np.array([vertex.y(), vertex.x()])

        if "camera-galvos" in self.dict_transformations.keys():
            vertices_transformed = transform(
                vertices, self.dict_transformations["camera-galvos"]
            )
        else:
            vertices_transformed = vertices / 2048 * 5
            print("Warning; galvos not registered")

        OriginalImage = np.zeros((1000, 1000))

        filled_mask = polygon2mask((1000, 1000), (vertices_transformed + 5) * 100)

        scanning_voltage = 5
        points_per_contour = 500
        sampling_rate = 50000

        contourScanningSignal = ProcessImage.mask_to_contourScanning_DAQsignals(
            filled_mask,
            OriginalImage,
            scanning_voltage,
            points_per_contour,
            sampling_rate,
            repeats=1,
        )

        readinchan = []
        digitalcontainer_array = np.zeros(0, dtype=tp_digital)  # len(...) = 0

        self.galvoThread = pmtimagingTest_contour()

        self.galvoThread.setWave_contourscan(
            sampling_rate, contourScanningSignal, points_per_contour
        )

        # self.galvothread = execute_analog_readin_optional_digital_thread()

        # self.galvothread.set_waves(samplingrate, contourScanningSignal,\
        # digitalcontainer_array, readinchan)
        # self.galvothread.start()
        # print('thread started')
        # self.galvothread = DAQmission()
        # self.galvothread.runWaveforms('DAQ', 4000, self.analogcontainer_array, digitalcontainer_array, readinchan)

    def start_galvo_scan(self):
        if self.ui_widget.galvosScanStartSwitch.isChecked():
            self.galvoThread.start()
        else:
            self.galvoThread.aboutToQuitHandler()

    # def createMaskSingleROI(self, vertices):
    #     if self.ui_widget.selectionMode == 'polygonMode':
    #         flag_fill_contour = self.ui_widget.polygonFillContourButton.isChecked()
    #         contour_thickness = self.ui_widget.polygonThicknessSpinBox.value()
    #     else:
    #         flag_fill_contour = self.ui_widget.freehandFillContourButton.isChecked()
    #         contour_thickness = self.ui_widget.freehandThicknessSpinBox.value()

    #     if flag_fill_contour:
    #         return polygon2mask((1024, 768), vertices)
    #     else:
    #         mask = np.zeros((1024,768))
    #         mask[polygon_perimeter(vertices[:,0], vertices[:,1], (1024,768))] = 1

    #         for _ in range(contour_thickness):
    #             mask = binary_dilation(binary_dilation(mask))
    #         return mask

    def createMask(self):
        if not self.flag_registered:
            self.ui_widget.normalOutputWritten(
                "Warning: Camera and DMD not registered!"
            )

        if self.ui_widget.drawing_laser == "640":
            roi_list = self.ui_widget.roi_list_640
            view = self.ui_widget.selection_view_640

        elif self.ui_widget.drawing_laser == "532":
            roi_list = self.ui_widget.roi_list_532
            view = self.ui_widget.selection_view_532
        else:
            roi_list = self.ui_widget.roi_list_488
            view = self.ui_widget.selection_view_488

        if self.ui_widget.selectionMode == "polygonMode":
            flag_fill_contour = self.ui_widget.polygonFillContourButton.isChecked()
            contour_thickness = self.ui_widget.polygonThicknessSpinBox.value()
        else:
            flag_fill_contour = self.ui_widget.freehandFillContourButton.isChecked()
            contour_thickness = self.ui_widget.freehandThicknessSpinBox.value()

        if self.ui_widget.selectionMode == "polygonMode":
            flag_invert_mode = self.ui_widget.polygonInvertMaskButton.isChecked()
        else:
            flag_invert_mode = self.ui_widget.freehandInvertMaskButton.isChecked()

        width = 2048 / view.getView().viewRect().width()
        height = 2048 / view.getView().viewRect().height()

        list_of_rois = []
        list_of_rois_transformed = []
        for roi in roi_list:
            roi_handle_positions = roi.getLocalHandlePositions()

            for idx, pos in enumerate(roi_handle_positions):
                roi_handle_positions[idx] = view.getView().mapToView(pos[1])

            num_vertices = len(roi_handle_positions)
            vertices = np.zeros([num_vertices, 2])

            for idx, vertex in enumerate(roi_handle_positions):
                vertices[idx, :] = np.array([vertex.x(), vertex.y()])

            vertices[:, 0] *= width
            vertices[:, 1] *= height

            laser = self.ui_widget.drawing_laser
            if "camera-dmd-" + laser in self.dict_transformations.keys():
                mask_transformed = np.zeros((1024, 768))
                vertices_transformed = transform(
                    vertices, self.dict_transformations["camera-dmd-" + laser]
                )
                list_of_rois_transformed.append(vertices_transformed)
            else:
                list_of_rois_transformed.append(vertices)
                print("Warning: not registered")

            vertices[:, 0] = vertices[:, 0] / 2048 * 1024
            vertices[:, 1] = vertices[:, 1] / 2048 * 768

            list_of_rois.append(vertices)

        self.mask_transformed[
            self.ui_widget.drawing_laser
        ] = ProcessImage.CreateBinaryMaskFromRoiCoordinates(
            list_of_rois_transformed,
            fill_contour=flag_fill_contour,
            contour_thickness=contour_thickness,
            invert_mask=flag_invert_mode,
        )
        self.mask[
            self.ui_widget.drawing_laser
        ] = ProcessImage.CreateBinaryMaskFromRoiCoordinates(
            list_of_rois,
            fill_contour=flag_fill_contour,
            contour_thickness=contour_thickness,
            invert_mask=flag_invert_mode,
        )

        self.save_mask()
        self.flag_mask_created = True
        self.ui_widget.update_buttons()

    # def createMask(self):
    #     """
    #     This function organizes the mask creation, depending on the mask settings.
    #     The function will preview the mask in the ROI window.
    #     """

    #     if not self.flag_registered:
    #         self.ui_widget.normalOutputWritten("Warning: Camera and DMD not registered!")

    #     if self.ui_widget.drawing_laser == '640':
    #         roi_list = self.ui_widget.roi_list_640
    #         view = self.ui_widget.selection_view_640

    #     elif self.ui_widget.drawing_laser == '532':
    #         roi_list = self.ui_widget.roi_list_532
    #         view = self.ui_widget.selection_view_532
    #     else:
    #         roi_list = self.ui_widget.roi_list_488
    #         view = self.ui_widget.selection_view_488

    #     if self.ui_widget.selectionMode == 'polygonMode':
    #         flag_invert_mode = self.ui_widget.polygonInvertMaskButton.isChecked()
    #     else:
    #         flag_invert_mode = self.ui_widget.freehandInvertMaskButton.isChecked()

    #     width = 2048/view.getView().viewRect().width()
    #     height = 2048/view.getView().viewRect().height()

    #     mask = np.zeros((1024, 768))

    #     for roi in roi_list:
    #         roi_handle_positions = roi.getLocalHandlePositions()

    #         for idx, pos in enumerate(roi_handle_positions):
    #             roi_handle_positions[idx] = view.getView().mapToView(pos[1])

    #         num_vertices = len(roi_handle_positions)
    #         vertices = np.zeros([num_vertices,2])

    #         for idx, vertex in enumerate(roi_handle_positions):
    #             vertices[idx,:] = np.array([vertex.x(), vertex.y()])

    #         vertices[:,0] *= width
    #         vertices[:,1] *= height

    #         laser = self.ui_widget.drawing_laser
    #         if 'camera-dmd-'+laser in self.dict_transformations.keys():
    #             mask_transformed = np.zeros((1024, 768))
    #             vertices_transformed = transform(vertices, self.dict_transformations['camera-dmd-'+laser])
    #             mask_transformed += self.createMaskSingleROI(vertices_transformed)
    #         else:
    #             print('Warning: not registered')

    #         vertices[:,0] = vertices[:,0] / 2048*1024
    #         vertices[:,1] = vertices[:,1] / 2048*768

    #         mask += self.createMaskSingleROI(vertices)

    #     mask = (mask > 0)*1
    #     if self.flag_registered:
    #         mask_transformed = (mask_transformed > 0)*1

    #     if flag_invert_mode:
    #         mask = 1 - mask
    #         if self.flag_registered:
    #             mask_transformed= 1 - mask_transformed

    #     self.mask[self.ui_widget.drawing_laser] = mask
    #     if self.flag_registered:
    #         self.mask_transformed[self.ui_widget.drawing_laser] = np.transpose(mask_transformed)
    #     else:
    #         self.mask_transformed[self.ui_widget.drawing_laser] = np.transpose(mask)

    #     self.save_mask()
    #     self.flag_mask_created = True
    #     self.ui_widget.update_buttons()

    def get_active_laser(self):
        """
        Retrieve which laser is active. If multiple lasers are active, use the most
        intens laser for registration.
        """
        laser_max_intensity = "640"
        max_intensity = 0
        for laser, state in self.lasers_status.items():
            value = state[1]
            if value > max_intensity:
                max_intensity = value
                laser_max_intensity = laser

        return laser_max_intensity

    ### REWRITE GALVO REGISTRATOR CLASS
    def galvos_register(self):
        self.registrator = RegistrationThread(self)
        self.registrator.set_device_to_register("galvos")
        self.registrator.start()
        self.registrator.sig_finished_registration.connect(self.registration_finished)

    def dmd_register(self, lasers_to_register):
        self.ui_widget.sig_start_registration.emit()

        self.flag_registrating = True
        self.ui_widget.update_buttons()

        self.DMD_Registration_thread = threading.Thread(
            target=lambda: self.dmd_register_func(lasers_to_register)
        )
        self.DMD_Registration_thread.start()
        self.DMD_Registration_thread.sig_finished_registration.connect(
            self.registration_finished
        )

    def dmd_register_func(self, lasers_to_register):
        thread = Registrator.DMDRegistrator()
        for laser, idx in enumerate(lasers_to_register):
            transformation = thread.registration(laser)

        self.dict_transformations.update(dict_transformations)

        self.ui_widget.normalOutputWritten("Registration finished succesfully")

        self.ui_widget.sig_finished_registration.emit()
        self.flag_registered = True
        self.flag_registrating = False
        self.flag_projecting = False
        self.ui_widget.update_buttons()

    def dmd_register1(self, lasers_to_register):
        self.ui_widget.sig_start_registration.emit()

        self.flag_registrating = True
        self.ui_widget.update_buttons()

        self.regthread = RegistrationThread(self, lasers_to_register)

        self.regthread.set_device_to_register("dmd")
        self.regthread.start()
        self.regthread.sig_finished_registration.connect(self.registration_finished)

    def manual_registration(self):
        self.manual_registration_window = ManualRegistration.ManualRegistrationWindow()

    # def registration_finished(self, dict_transformations):
    #     self.dict_transformations.update(dict_transformations)

    #     self.ui_widget.normalOutputWritten("Registration finished succesfully")

    #     self.ui_widget.sig_finished_registration.emit()
    #     self.flag_registered = True
    #     self.flag_registrating = False
    #     self.flag_projecting = False
    #     self.ui_widget.update_buttons()

    def check_mask_format_valid(self, mask):
        if len(mask.shape) == 3:
            self.ui_widget.normalOutputWritten(
                "Image is stack; make sure to load a binary image; for now max projection used"
            )
            mask = np.max(mask, axis=2)

        if mask.shape[0] == 1024 and mask.shape[1] == 768:
            mask = mask.transpose()

        elif mask.shape[0] != 768 or mask.shape[1] != 1024:
            self.ui_widget.normalOutputWritten(
                "Image has shape "
                + str(mask.shape[0])
                + "x"
                + str(mask.shape[1])
                + "; should be 768x1024"
            )
            return False, None

        return True, mask

    def loadFile(self):
        """
        Load file from path and put image to mask display.
        """
        try:
            mask = plt.imread(self.loadFileName)
        except:
            self.ui_widget.normalOutputWritten("Invalid file path or file")
            return

        valid, valid_mask = self.check_mask_format_valid(mask)
        if not valid:
            return
        else:
            self.mask[self.ui_widget.drawing_laser] = self.mask_transformed[
                self.ui_widget.drawing_laser
            ] = valid_mask

        self.flag_mask_created = True
        self.ui_widget.update_buttons()

    def loadFolder(self):
        """
        Load files from folder using path and save frames in multidimensional array.
        """

        if not self.loadFolderName:
            return

        list_dir_raw = sorted(os.listdir(self.loadFolderName))

        list_dir = [file for file in list_dir_raw if file[-3:] in ["png", "jpg"]]
        list_nr = len(list_dir)
        img = np.zeros([768, 1024, list_nr])
        for i in range(list_nr):
            single_mask = plt.imread(self.loadFolderName + "/" + list_dir[i])
            valid, valid_single_mask = self.check_mask_format_valid(single_mask)
            if not valid:
                return
            else:
                img[:, :, i] = valid_single_mask

        self.mask_transformed[self.ui_widget.drawing_laser] = img
        if self.ui_widget.invertMovieButton.isChecked():
            self.mask_transformed[self.ui_widget.drawing_laser] = (
                1 - self.mask_transformed[self.ui_widget.drawing_laser]
            )

        self.flag_mask_created = True
        self.ui_widget.update_buttons()


def transform(r, A):
    """
    This function takes points as input and returns the
    transformed points.

    r = np.array([[1,1], [1, 2], [1,3]])

    """

    if r.ndim == 1:
        Q = createTransformationMatrix(r)

        if Q is None:
            return

        return np.squeeze(np.reshape(np.dot(Q, A), (-1, 2), order="F"))

    else:
        num_points = r.shape[0]

    transformed_points = np.zeros([num_points, 2])

    for i in range(num_points):

        Q = createTransformationMatrix(r[i, :])

        if Q is None:
            return

        transformed_points[i, :] = np.squeeze(np.dot(Q, A))
    return np.reshape(transformed_points, (-1, 2), order="F")


def transform(r, c):
    x = r[0, :]
    y = r[1, :]

    # Maybe this functio nrequires meshgrid inputs
    return np.polynomial.polynomial.polyval2d(x, y, np.transpose(c))


def createTransformationMatrix(q, order=1):
    if len(q.shape) == 1:
        Qx = np.array([1, 0, q[0], q[1]])
        Qy = np.hstack((0, 1, np.zeros(2 * order), q[0], q[1]))

        for i in range(2, order + 1):
            Qx = np.hstack((Qx, q[0] ** i, q[1] ** i))
            Qy = np.hstack((Qy, q[0] ** i, q[1] ** i))

        Qx = np.hstack((Qx, np.zeros(2 * order)))
    else:
        print("Function takes only one point at a time")
        return

    return np.vstack((Qx, Qy))


if __name__ == "__main__":

    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        mainwin = CoordinatesWidgetUI()
        mainwin.show()
        app.exec_()

    run_app()
