#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:38:53 2020

@author: Izak de Heer
"""

import importlib.resources
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
from qtpy import QtGui, QtWidgets
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)
from skimage.draw import polygon2mask

from ..GalvoWidget.pmt_thread import pmtimagingTest_contour
from ..ImageAnalysis.ImageProcessing import ProcessImage
from ..StylishQT import roundQGroupBox
from . import CoordinateTransformations, Registration, Registrator


class GalvoWidget(QWidget):
    sig_request_mask_coordinates = Signal()
    sig_start_registration = Signal()
    sig_finished_registration = Signal()

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.main_application = parent

        self.set_transformation_saving_location(
            importlib.resources.files(Registration)
        )

        self.init_gui()

    def init_gui(self):
        layout = QGridLayout()

        # self.setFixedSize(240,200)
        self.setFixedHeight(200)
        self.box = roundQGroupBox()
        self.box.setTitle("Galvo control")
        box_layout = QGridLayout()
        self.box.setLayout(box_layout)

        self.setLayout(layout)

        self.register_button = QPushButton("Register")

        self.create_voltage_button = QPushButton("Create voltage")

        self.points_per_contour_textbox = QLineEdit()
        self.points_per_contour_textbox.setValidator(QtGui.QIntValidator())
        self.points_per_contour_textbox.setText("100")

        self.sampling_rate_textbox = QLineEdit()
        self.sampling_rate_textbox.setValidator(QtGui.QIntValidator())
        self.sampling_rate_textbox.setText("50000")

        self.scan_button = QPushButton("Start scan")
        self.scan_button.clicked.connect(self.scan)

        self.register_button.clicked.connect(self.register)
        self.create_voltage_button.clicked.connect(
            self.request_mask_coordinates
        )

        box_layout.addWidget(self.register_button, 1, 0)
        box_layout.addWidget(QLabel("Points per contour:"))
        box_layout.addWidget(self.points_per_contour_textbox, 2, 1)
        box_layout.addWidget(QLabel("Sampling rate:"), 3, 0)
        box_layout.addWidget(self.sampling_rate_textbox, 3, 1)
        box_layout.addWidget(self.create_voltage_button, 4, 0)
        box_layout.addWidget(self.scan_button, 4, 1)

        layout.addWidget(self.box)

        self.open_latest_transformation()

    def register(self):
        self.sig_start_registration.emit()

        self.registrator = Registrator.GalvoRegistrator()
        self.transform = self.registrator.registration()
        self.save_transformation()

        self.sig_finished_registration.emit()

    def request_mask_coordinates(self):
        self.sig_request_mask_coordinates.emit()

    def receive_mask_coordinates(self, sig_from_CoordinateWidget):
        """
        Receive signal from CoordinateWidget
        """
        # !!! need to adapt to multiple frames!
        for each_roi_index in range(len(sig_from_CoordinateWidget)):
            list_of_rois = sig_from_CoordinateWidget[each_roi_index][0]
            list_of_rois = self.transform_coordinates(list_of_rois)

            self.create_voltage_signal(list_of_rois)

    def transform_coordinates(self, list_of_rois):
        """
        Given list of roi positions in camera image, transform into corrseponding
        voltage positions.

        Parameters
        list_of_rois : list
            DESCRIPTION.

        Returns
        new_list_of_rois : TYPE
            DESCRIPTION.

        """
        new_list_of_rois = []
        for roi in list_of_rois:
            new_list_of_rois.append(
                np.flip(
                    CoordinateTransformations.transform(
                        np.flip(roi), self.transform
                    )
                )
            )

        return new_list_of_rois

    def create_voltage_signal(self, list_of_rois):
        filled_mask = OriginalImage = np.zeros((1000, 1000))

        for roi in list_of_rois:
            filled_mask += polygon2mask((1000, 1000), (roi + 5) * 100)

        filled_mask = (filled_mask > 0).astype(int).transpose()
        fig, axs = plt.subplots(1, 1)
        axs.imshow(filled_mask)

        scanning_voltage = 5
        points_per_contour = int(self.points_per_contour_textbox.text())
        sampling_rate = int(self.sampling_rate_textbox.text())

        contourScanningSignal = (
            ProcessImage.mask_to_contourScanning_DAQsignals(
                filled_mask,
                OriginalImage,
                scanning_voltage,
                points_per_contour,
                sampling_rate,
                repeats=1,
            )
        )

        contourScanningSignal = np.vstack(
            (contourScanningSignal[0][0], contourScanningSignal[1][0])
        )

        self.galvoThread = pmtimagingTest_contour()
        self.galvoThread.setWave_contourscan(
            sampling_rate, contourScanningSignal, points_per_contour
        )

    def scan(self):
        if self.scan_button.text() == "Start scan":
            self.galvoThread.start()
            self.scan_button.setText("Stop scan")
        else:
            self.galvoThread.aboutToQuitHandler()
            self.scan_button.setText("Start scan")

    def set_transformation_saving_location(self, traversable):
        self.transformation_location = traversable

    def save_transformation(self):
        size = self.transform.shape[0]
        traversable = self.transformation_location.joinpath(
            "galvo_transformation"
        )
        with importlib.resources.as_file(traversable) as path:
            np.savetxt(path.as_posix(), np.reshape(self.transform, (-1, size)))

    def open_latest_transformation(self):
        traversable = self.transformation_location.joinpath(
            "galvo_transformation"
        )
        with importlib.resources.as_file(traversable) as path:
            transform = np.loadtxt(path.as_posix())

        self.transform = np.reshape(transform, (transform.shape[1], -1, 2))
        logging.info("Transform for galvos loaded:")
        logging.info(self.transform[:, :, 0])
        logging.info(self.transform[:, :, 1])


if __name__ == "__main__":

    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        mainwin = GalvoWidget()
        mainwin.show()
        app.exec_()

    run_app()
