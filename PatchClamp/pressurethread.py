# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:29:20 2021

@author: TvdrBurgt
"""


import time
import serial
import logging
import numpy as np
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread, QMutex


class PressureThread(QThread):
    """ Pressure control through serial communication
    This class is for controlling the Pressure Controller.
    """
    measurement = pyqtSignal(np.ndarray)
    
    def __init__(self, address='COM4', baud=9600):
        self.mutex = QMutex()
        
        # QThread attributes
        super().__init__()
        self.isrunning = False
        self.moveToThread(self)
        self.started.connect(self.measure)
        
        # Serial attributes
        self.port = address     # COM port micromanipulator is connected to
        self.baudrate = baud    # Baudrate of the micromanipulator
        self.ENDOFLINE = '\n'   # Carriage return
        self.controller = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=1)
    
    def stop(self):
        self.isrunning = False
        self.quit()
        self.wait()
    
    def set_pressure(self, pressure):
        """
        Sets the pressure immediately, and does not depend on the measurement
        thread.
        """
        command = "P %d" % pressure + self.ENDOFLINE
        
        # Encode the command to ascii and send to the device
        self.controller.write(command.encode('ascii'))
        
    @pyqtSlot()
    def measure(self):
        logging.info('pressure thread started')
        
        self.isrunning = True
        while self.isrunning:
            # Read pressure controller output
            response = self.controller.read_until(self.ENDOFLINE.encode('ascii'))
            try:
                response = response.decode('utf-8')
            except:
                response = ""
            response = response.split()
            
            # If output is from the pressuresensors we emit their value
            if len(response) > 0:
                if response[0] == "PS":
                    PS1 = float(response[1])
                    PS2 = float(response[2])
                    self.measurement.emit(np.array([PS1, PS2]))
            
            # Not really necessary but better to safe computation power
            QThread.msleep(25)
        
        # Set pressure back to ATM and close the serial port
        self.set_pressure(0)
        self.controller.close()
            
        logging.info('pressure thread stopped')


# class PressureThread(QThread):
#     """ Pressure control through serial communication
#     This class is for controlling the Pressure Controller.
#     """
#     measurement = pyqtSignal(np.ndarray)
    
#     def __init__(self, address='COM4', baud=9600):
#         self.port = address     # COM port micromanipulator is connected to
#         self.baudrate = baud    # Baudrate of the micromanipulator
#         self.ENDOFLINE = '\n'   # Carriage return
        
#         self.pressure_offset = 0
        
#         # QThread attributes
#         super().__init__()
#         self.isrunning = False
#         self.moveToThread(self)
#         self.started.connect(self.measure)
    
#     def stop(self):
#         self.isrunning = False
#         self.quit()
#         self.wait()
    
#     def set_pressure(self, target_pressure):
#         self.pressure_offset = target_pressure
    
#     @pyqtSlot()
#     def measure(self):
#         logging.info('pressure thread started')
        
#         self.isrunning = True
#         while self.isrunning:
#             output = self.pressure_offset + np.random.rand(2)*10-5
#             self.measurement.emit(np.array([output[0], output[1]]))
#             QThread.msleep(50)
        
#         logging.info('pressure thread stopped')
