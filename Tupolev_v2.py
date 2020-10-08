#c -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 20:54:40 2019

@author: xinmeng
    ============================== ==============================================
    
    For general experiments in Dr. Daan's lab ゴゴゴ ゴ ゴ ゴ  ゴ  ゴ
    ============================== ==============================================
    == Widget descriptions ==
    
    - PatchClamp.ui_patchclamp_sealtest: The GUI for patch clamp.
    - NIDAQ.Waveformer_for_screening: The GUI for configuring and executing waveforms in National Instrument Data Acquisition (DAQ) device.
    - GalvoWidget.PMTWidget: For PMT scanning imaging.
    - ImageAnalysis.AnalysisWidget: Data Analysis widget.
    - SampleStageControl.StageMoveWidget: The GUI for sample stage movement control.
    - NIDAQ.AOTFWidget: To control AOTF, which is controlled by NI-daq.
    - ThorlabsFilterSlider.FilterSliderWidget: Controller for Thorlabs filter slider.
    - PI_ObjectiveMotor.ObjMotorWidget: Widget for objective motor control.
    - CoordinatesManager.CoordinatesWidget: Widget to create mask based on widefield image. Project mask with DMD or galvos
    ============================== ==============================================
"""
from __future__ import division
import os
import sys
sys.path.append('../')
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPoint, QRect, QObject, QSize
from PyQt5.QtGui import QColor, QPen, QPixmap, QIcon, QTextCursor, QFont, QPalette, QBrush, QImage

from PyQt5.QtWidgets import (QWidget, QButtonGroup, QLabel, QSlider, QSpinBox, QDoubleSpinBox, QGridLayout, QPushButton, QGroupBox, 
                             QLineEdit, QVBoxLayout, QHBoxLayout, QComboBox, QMessageBox, QTabWidget, QCheckBox, QRadioButton, 
                             QFileDialog, QProgressBar, QTextEdit, QStyleFactory, QGraphicsDropShadowEffect)

import pyqtgraph as pg
import StylishQT

import PatchClamp.ui_patchclamp_sealtest
import NIDAQ.WaveformWidget
import GalvoWidget.PMTWidget
import ImageAnalysis.AnalysisWidget
import SampleStageControl.StageMoveWidget
import NIDAQ.AOTFWidget
import NIDAQ.DAQoperator
import ThorlabsFilterSlider.FilterSliderWidget
import PI_ObjectiveMotor.ObjMotorWidget
import ThorlabsKCube.KCubeWidget
import InsightX3.TwoPhotonLaserUI
import Weather_GUI
import Evolution_GUI_v7

import pyqtgraph.console
import HamamatsuCam.HamamatsuUI
import CoordinatesManager.CoordinateWidget2

#Setting graph settings
#"""
#pg.setConfigOption('background', 'w')
#pg.setConfigOption('foreground', 'k')
#pg.setConfigOption('useOpenGL', True)
#pg.setConfigOption('leftButtonPan', False)
#""" 
#class EmittingStream(QObject): #https://stackoverflow.com/questions/8356336/how-to-capture-output-of-pythons-interpreter-and-show-in-a-text-widget
#    textWritten = pyqtSignal(str)
#    def write(self, text):
#        self.textWritten.emit(str(text)) # For updating notice from console.   

class Mainbody(QWidget):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Set directory to current folder. Specific command depends on platform, 
        # i.e. first command gives error in Linux and second command gives error
        # on Windows.
        try:
            os.chdir(os.path.dirname(sys.argv[0]))
        except:
            pass
            # os.chdir(sys.path[0])
                
#        os.chdir(os.path.dirname(sys.argv[0]))# Set directory to current folder.
            
        self.setWindowIcon(QIcon('./Icons/Icon.png'))
        self.setFont(QFont("Arial"))
#        blur_effect = QtWidgets.QGraphicsBlurEffect(blurRadius=5)
#        self.setGraphicsEffect(blur_effect)
        self.OC = 0.1
        
        #----------------------------------------------------------------------
        #----------------------------------GUI---------------------------------
        #----------------------------------------------------------------------
        self.setMinimumSize(1600,1080)
        self.setWindowTitle("Fiumicino")
        self.layout = QGridLayout(self)
        """
        # =============================================================================
        #         GUI for right tabs panel-Creating instances of each widget showing on right side tabs.
        # =============================================================================
        """
        self.tabs = QTabWidget()
        self.Galvo_WidgetInstance = GalvoWidget.PMTWidget.PMTWidgetUI()
        self.Waveformer_WidgetInstance = NIDAQ.WaveformWidget.WaveformGenerator()
        self.PatchClamp_WidgetInstance = PatchClamp.ui_patchclamp_sealtest.PatchclampSealTestUI()
        self.Analysis_WidgetInstance = ImageAnalysis.AnalysisWidget.AnalysisWidgetUI()
        self.Coordinate_WidgetInstance = CoordinatesManager.CoordinateWidget2.CoordinatesWidgetUI(self)
        
        #--------------Add tab widgets-------------------
        self.tabs.addTab(self.Galvo_WidgetInstance,"PMT imaging")
        self.tabs.addTab(self.Waveformer_WidgetInstance,"Waveform")
        self.tabs.addTab(self.PatchClamp_WidgetInstance,"Patch clamp")    
        self.tabs.addTab(self.Coordinate_WidgetInstance, "Coordinates")
        self.tabs.addTab(self.Analysis_WidgetInstance,"Image analysis")
        # =============================================================================
        
        self.savedirectory = os.path.join(os.path.expanduser("~"), "Desktop") #'M:/tnw/ist/do/projects/Neurophotonics/Brinkslab/Data'
        
        """
        # =============================================================================
        #         GUI for left panel.
        # =============================================================================        
        """
        # =============================================================================
        #         GUI for set directory
        # =============================================================================
        setdirectoryContainer = StylishQT.roundQGroupBox(title="Set directory")
        self.setdirectorycontrolLayout = QGridLayout()        
        
        self.saving_prefix = ''
        self.savedirectorytextbox = QLineEdit(self)
        self.savedirectorytextbox.setPlaceholderText('Saving directory')
        self.savedirectorytextbox.returnPressed.connect(self.update_saving_directory)
        self.setdirectorycontrolLayout.addWidget(self.savedirectorytextbox, 0, 1)
        
        self.prefixtextbox = QLineEdit(self)
        self.prefixtextbox.setPlaceholderText('Prefix')
        self.prefixtextbox.returnPressed.connect(self.set_prefix)
        self.setdirectorycontrolLayout.addWidget(self.prefixtextbox, 0, 0)
        
        #self.setdirectorycontrolLayout.addWidget(QLabel("Saving prefix:"), 0, 0)
        
        self.toolButtonOpenDialog = QtWidgets.QPushButton()
        self.toolButtonOpenDialog.setIcon(QIcon('./Icons/Browse.png')) 
        self.toolButtonOpenDialog.setObjectName("toolButtonOpenDialog")
        self.toolButtonOpenDialog.clicked.connect(self.set_saving_directory)
        
        self.setdirectorycontrolLayout.addWidget(self.toolButtonOpenDialog, 0, 2)
        
        setdirectoryContainer.setLayout(self.setdirectorycontrolLayout)
        setdirectoryContainer.setMaximumHeight(70)
        setdirectoryContainer.setMaximumWidth(420)
        
        self.layout.addWidget(setdirectoryContainer, 0, 0, 1, 2)
        
        # =============================================================================
        #         GUI for toolbox
        # =============================================================================
        toolboxContainer = StylishQT.roundQGroupBox(title="Toolbox")
        self.toolboxContainerlLayout = QGridLayout()
        
        self.shutter2PButton = StylishQT.checkableButton(Icon_path = './Icons/shutter.png')
        self.shutter2PButton.clicked.connect(self.shutter2Paction)        
        self.shutter2PButton.setToolTip("Open/Close InsightX3 shutter")
        
        self.LEDButton = StylishQT.checkableButton(Icon_path = './Icons/LED.png')
        self.LEDButton.clicked.connect(self.LEDaction)
        self.LEDButton.setToolTip("Turn on/off LED")
        
        self.shutter2PButton.setGraphicsEffect(QGraphicsDropShadowEffect(blurRadius=3, xOffset=2, yOffset=2))
        self.LEDButton.setGraphicsEffect(QGraphicsDropShadowEffect(blurRadius=3, xOffset=2, yOffset=2))
        
        self.toolboxContainerlLayout.addWidget(self.shutter2PButton, 0, 1)
        self.toolboxContainerlLayout.addWidget(self.LEDButton, 0, 2)
        
        toolboxContainer.setLayout(self.toolboxContainerlLayout)

        self.layout.addWidget(toolboxContainer, 1, 1)
        
        # =============================================================================
        #         GUI for weather
        # =============================================================================
        self.layout.addWidget(Weather_GUI.WeatherUI(), 1, 0)
        
        # =============================================================================
        #         GUI for Thorlabs motor
        # =============================================================================
        self.KCubeWidgetInstance = SampleStageControl.StageMoveWidget.StageWidgetUI()
        self.layout.addWidget(self.KCubeWidgetInstance, 4, 0, 1, 2)
        
        # =============================================================================
        #         GUI for sample stage
        # =============================================================================
        self.StageMoveWidgetInstance = ThorlabsKCube.KCubeWidget.KCubeWidgetUI()
        self.layout.addWidget(self.StageMoveWidgetInstance, 6, 0, 1, 2)

        # =============================================================================
        #         GUI for AOTF
        # =============================================================================             
        self.AOTFWidgetInstance = NIDAQ.AOTFWidget.AOTFWidgetUI()
        self.layout.addWidget(self.AOTFWidgetInstance, 5, 0, 1, 2)

        # =============================================================================
        #         GUI for fliter silder
        # =============================================================================        
        FilterSliderWidgetInstance = ThorlabsFilterSlider.FilterSliderWidget.FilterSliderWidgetUI()
        self.layout.addWidget(FilterSliderWidgetInstance, 8, 0, 1, 2)    

        # =============================================================================
        #         GUI for objective motor
        # =============================================================================        
        ObjMotorInstance = PI_ObjectiveMotor.ObjMotorWidget.ObjMotorWidgetUI()
        self.layout.addWidget(ObjMotorInstance, 7, 0, 1, 2)         

        # =============================================================================
        #         GUI for camera button       
        # =============================================================================
        self.open_cam = StylishQT.FancyPushButton(55, 25, color1=(255,153,255), color2=(204,208,255))
        self.open_cam.setIcon(QIcon('./Icons/Hamamatsu.png'))
        self.open_cam.setToolTip("Open camera widget")
        self.open_cam.setIconSize(QSize(100, 100))
        self.open_cam.clicked.connect(self.open_camera)
        self.layout.addWidget(self.open_cam, 2, 0, 1, 2)
        self.open_cam.setGraphicsEffect(QGraphicsDropShadowEffect(blurRadius=3, xOffset=2, yOffset=2))
        # =============================================================================
        #         GUI for Insight X3      
        # =============================================================================
        self.open_Insight = StylishQT.FancyPushButton(50, 50, color1=(176,224,230), color2=(135,206,250))
        self.open_Insight.setIcon(QIcon("./Icons/spectra_physics.png"))
        self.open_Insight.setToolTip("Open 2-p laser widget")
        self.open_Insight.setIconSize(QSize(40, 40))
        self.open_Insight.clicked.connect(self.open_Insight_UI)
        self.layout.addWidget(self.open_Insight, 3, 0, 1, 1)
        self.open_Insight.setGraphicsEffect(QGraphicsDropShadowEffect(blurRadius=3, xOffset=2, yOffset=2))
        # =============================================================================
        #         GUI for evolution screening     
        # =============================================================================
        self.open_screening_button = StylishQT.FancyPushButton(50, 50, color1=(245,245,220), color2=(255,228,196))
        self.open_screening_button.setIcon(QIcon("./Icons/screening.png"))
        self.open_screening_button.setToolTip("Open screening widget")
        self.open_screening_button.setIconSize(QSize(45, 45))
        self.open_screening_button.clicked.connect(self.open_screening)
        self.layout.addWidget(self.open_screening_button, 3, 1, 1, 1)
        self.open_screening_button.setGraphicsEffect(QGraphicsDropShadowEffect(blurRadius=3, xOffset=2, yOffset=2))
        # =============================================================================
        #         Console massage    
        # =============================================================================
        self.console_text_edit = QTextEdit()
        self.console_text_edit.setFontItalic(True)
        self.console_text_edit.setPlaceholderText('Notice board from console.')
        self.console_text_edit.setFixedHeight(200)
        self.layout.addWidget(self.console_text_edit, 9, 0, 1, 2)
        
        #**************************************************************************************************************************************        
        #self.setLayout(pmtmaster)
            
        self.layout.addWidget(self.tabs, 0, 2, 10, 4)
        self.setLayout(self.layout)
        
        # =============================================================================
        #         Establishing communication between widgets.
        # =============================================================================
        self.Galvo_WidgetInstance.SignalForContourScanning.connect(self.PassVariable_GalvoWidget_to_Waveformer)
        self.Galvo_WidgetInstance.MessageBack.connect(self.normalOutputWritten)
        
        self.Coordinate_WidgetInstance.sig_start_registration.connect(lambda: self.AOTFWidgetInstance.set_registration_mode(True))
        self.Coordinate_WidgetInstance.sig_finished_registration.connect(lambda: self.AOTFWidgetInstance.set_registration_mode(False))
        # self.Coordinate_WidgetInstance.sig_control_laser.connect(self.AOTFWidgetInstance.control_for_registration)
        # self.Coordinate_WidgetInstance.sig_console_print.connect(self.normalOutputWritten)
        
        # self.AOTFWidgetInstance.sig_lasers_status_changed.connect(self.Coordinate_WidgetInstance.lasers_status_changed)
        
        self.Analysis_WidgetInstance.MessageBack.connect(self.normalOutputWritten)
        self.Analysis_WidgetInstance.Cellselection_DMD_mask_contour.connect(self.Coordinate_WidgetInstance.DMDWidget.receive_mask_coordinates)
        '''
        ***************************************************************************************************************************************
        ************************************************************END of GUI*****************************************************************
        '''
    #%%
    def __del__(self):
        # Restore sys.stdout
        sys.stdout = sys.__stdout__
        
        '''
        ***************************************************************************************************************************************
        ************************************************************ Functions to pass variables across widges ********************************
        '''        
    def PassVariable_GalvoWidget_to_Waveformer(self, contour_point_number, Daq_sample_rate_pmt, time_per_contour, handle_viewbox_coordinate_x, handle_viewbox_coordinate_y):
        
        self.Waveformer_WidgetInstance.galvo_contour_label_1.setText("Points in contour: %.d" % contour_point_number)
        self.Waveformer_WidgetInstance.galvo_contour_label_2.setText("Sampling rate: %.d" % Daq_sample_rate_pmt)
        self.Waveformer_WidgetInstance.Daq_sample_rate_pmt = Daq_sample_rate_pmt
        self.Waveformer_WidgetInstance.handle_viewbox_coordinate_position_array_expanded_x = handle_viewbox_coordinate_x
        self.Waveformer_WidgetInstance.handle_viewbox_coordinate_position_array_expanded_y = handle_viewbox_coordinate_y
        self.Waveformer_WidgetInstance.time_per_contour = time_per_contour
        
#    def PassVariable_AnalysisWidget_to_DMDWidget(self, output_mask_from_Cellselection):
#        print('Mask from AnalysisWidget_to_DMDWidget')
#        
#        self.Coordinate_WidgetInstance.mask = output_mask_from_Cellselection
#        self.Coordinate_WidgetInstance.mask_view.setImage(output_mask_from_Cellselection)
    # =============================================================================
    #     Fucs for set directory
    # =============================================================================
    # Set the savedirectory and prefix of Waveform widget in syn.
    def set_saving_directory(self):
        self.savedirectory = str(QtWidgets.QFileDialog.getExistingDirectory())
        self.savedirectorytextbox.setText(self.savedirectory)
        self.Galvo_WidgetInstance.savedirectory = self.savedirectory        
        self.Waveformer_WidgetInstance.savedirectory = self.savedirectory        
        self.Analysis_WidgetInstance.savedirectory = self.savedirectory
        self.PatchClamp_WidgetInstance.saving_dir = self.savedirectory
        
        self.set_prefix()
    
    def update_saving_directory(self):
        self.savedirectory = str(self.savedirectorytextbox.text())
        
    def set_prefix(self):
        self.saving_prefix = str(self.prefixtextbox.text())

        self.Galvo_WidgetInstance.prefixtextboxtext = self.saving_prefix
        self.Waveformer_WidgetInstance.saving_prefix = self.saving_prefix
        
    def shutter2Paction(self):
        daq= NIDAQ.DAQoperator.DAQmission()
        # For 2P shutter
        if self.shutter2PButton.isChecked():
            daq.sendSingleDigital('2Pshutter', True)
        else:
            daq.sendSingleDigital('2Pshutter', False)
            
    def LEDaction(self):
        daq= NIDAQ.DAQoperator.DAQmission()
        # For LED
        if self.LEDButton.isChecked():
            daq.sendSingleDigital('LED', True)
        else:
            daq.sendSingleDigital('LED', False)        
    # =============================================================================
    #    Fucs for camera options
    # =============================================================================
    def open_camera(self):
        self.camWindow = HamamatsuCam.HamamatsuUI.CameraUI()
        self.camWindow.show()
        
        # Connect camera with DMD widget, so that snapped images are shown in 
        # DMD widget.
        self.camWindow.signal_SnapImg.connect(self.Coordinate_WidgetInstance.set_camera_image)
        
    def open_Insight_UI(self):
        self.open_Insight_UIWindow = InsightX3.TwoPhotonLaserUI.InsightWidgetUI()
        self.open_Insight_UIWindow.show()
        
    def open_screening(self):
        self.open_screening_UIWindow = Evolution_GUI_v7.Mainbody()
        self.open_screening_UIWindow.show()
    # =============================================================================
    #     Fucs for console display
    # =============================================================================
    def normalOutputWritten(self, text):
        """Append text to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        cursor = self.console_text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.console_text_edit.setTextCursor(cursor)
        self.console_text_edit.ensureCursorVisible()
        
    def closeEvent(self, event):
        QtWidgets.QApplication.quit()
        event.accept()
        
if __name__ == "__main__":
    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        QtWidgets.QApplication.setStyle(QStyleFactory.create('Fusion'))
        pg.setConfigOptions(imageAxisOrder='row-major')
        mainwin = Mainbody()
        mainwin.show()
        app.exec_()
    run_app()
    