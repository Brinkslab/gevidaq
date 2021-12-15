# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 18:21:18 2021

@author: TvdrBurgt
"""

import os
import json
import logging
import numpy as np
from PyQt5.QtCore import QObject, QThread

from PatchClamp.workers import Worker


class SmartPatcher(QObject):
    
    def __init__(self):
        # Default hardware constants
        self._pixel_size = 229.8                    # in nanometers
        self._image_size = [2048, 2048]             # dimension of FOV in pix
        self._pipette_orientation = 0               # in radians
        self._pipette_diameter = 16                 # in pixels (16=patchclamp, ??=cell-picking)
        self._rotation_angles = [0,0,0.043767]      # (alp,bet,gam) in radians
        self._focus_offset = 30                     # in micron above coverslip
        self.update_constants_from_JSON()           # rewrites above default constants
        
        # Algorithm constants
        self.save_directory = os.getcwd()+'\\PatchClamp\\feedback\\'
        self._operation_mode = 'Default'            # specifies worker mode
        self._pipette_coordinates_pair = np.array(  # [micromanipulator, camera]
            [[None,None,None], [None,None,None]])
        self._target_coordinates = np.array(        # [X, Y, Z] in pixels
            [None,None,None])
        self._resistance_reference = None           # in MΩ
        self.window_size_c = 200
        self.window_size_v = 200
        self.window_size_p = 200
        self.window_size_rc = 200
        self.n_c = 0
        self.n_v = 0
        self.n_p = 0
        self.n_rc = 0
        self._current = np.array([])
        self._voltage = np.array([])
        self._pressure = np.array([[],[]])
        self._resistance = np.array([])
        self._capacitance = np.array([])
        
        # Hardware devices
        self._camerathread = None
        self._sealtestthread = None
        self._pressurethread = None
        self._micromanipulator = None
        self._objectivemotor = None
        self._XYstage = None
        
        # Worker thread
        self.worker = Worker(self)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.thread.quit)
        
        # Emergency stop
        self.STOP = False
    
    def request(self, name, mode='default'):
        if self.STOP == True:
            logging.info('Emergency stop active')
        else:
            logging.info('Reqeusted algorithm: ' + name)
            if self.thread.isRunning() == True:
                logging.info('Thread is still running')
            else:
                # disconnect the started method to prevent double execution
                try:
                    self.thread.started.disconnect()
                except TypeError:
                    logging.info('Thread is free to use')
                    
                # workers can use operation modes for extra variable input
                self.operation_mode = mode
                
                # connected the started method to an executable algorithm
                if name == 'hardcalibration':
                    if self.camerathread == None or self.micromanipulator == None:
                        raise ValueError('Camera and/or micromanipulator not connected')
                    else:
                        self.thread.started.connect(self.worker.softcalibration)
                elif name == 'prechecks':
                    if self.sealtestthread == None or self.pressurethread == None:
                        raise ValueError('Patch amplifier and/or pressure controller not connected')
                    else:
                        self.thread.started.connect(self.worker.prechecks)
                elif name == 'softcalibration':
                    if self.camerathread == None or self.micromanipulator == None:
                        raise ValueError('Camera and/or micromanipulator not connected')
                    else:
                        self.thread.started.connect(self.worker.hardcalibration)
                elif name == 'autofocustip':
                    if self.camerathread == None or self.micromanipulator == None or self.objectivemotor == None:
                        raise ValueError('Camera, objective and/or micromanipulator not connected')
                    else:
                        self.thread.started.connect(self.worker.autofocus_tip)
                elif name == 'target2center':
                    if self.XYstage == None or np.array_equal(self.target_coordinates, [None,None,None]):
                        raise ValueError('XY stage not connected')
                    else:
                        self.thread.started.connect(self.worker.target2center)
                elif name == 'pipette2target':
                    if self.micromanipulator == None or np.array_equal(self.target_coordinates, [None,None,None]) or \
                        np.array_equal(self.pipette_coordinates_pair, [[None,None,None], [None,None,None]]):
                        raise ValueError('Target not selected, pipette tip not detected, and/or micromanipulator not connected')
                    else:
                        self.thread.started.connect(self.worker.pipette2target)
                elif name == 'gigaseal':
                    if self.sealtestthread == None or self.pressurethread == None or self.micromanipulator == None:
                        raise ValueError('Patch amplifier, pressure controller and/or micromanipulator not connected')
                    else:
                        self.thread.started.connect(self.worker.formgigaseal)
                elif name == 'breakin':
                    if self.sealtestthread == None or self.pressurethread == None:
                        raise ValueError('Sealtest and/or pressure controller not connected')
                    else:
                        self.thread.started.connect(self.worker.break_in)
                elif name == 'mockworker':
                    self.thread.started.connect(self.worker.mockworker)
                elif name == 'request_imagexygrid':
                    self.thread.started.connect(self.worker.request_imagexygrid)
                elif name == 'request_imagezstack':
                    self.thread.started.connect(self.worker.request_imagezstack)
                
                # start worker
                self.thread.start()
                
            logging.info('QThread isFinished: ' + str(self.thread.isFinished()))
            logging.info('QThread isRunning: ' + str(self.thread.isRunning()))
    
    def account4rotation(self, origin, target):
        """
        This function accounts for the misalignment of the micromanipulator w.r.t
        the camera FOV. The origin is the rotation point where the rotation
        matrix R rotates about.
        
        input:
            origin  = point of rotation (np.ndarray with shape (3,))
            target  = target coordinates (np.ndarray with shape (3,))
        
        output:
            newtarget   = rotated target coordinates (np.ndarray with shape (3,))
        """
        if isinstance(origin, np.ndarray) and isinstance(target, np.ndarray):
            if origin.shape == (3,) and target.shape == (3,):
                pass
            else:
                raise ValueError('origin and target should have shape (3,)')
        else:
            raise ValueError('origin and target should be numpy.ndarray')
        
        return self.R @ np.subtract(target,origin) + origin

    def update_constants_from_JSON(self):
        # read json file with autopatcher constants and update them in backend
        try:
            with open("PatchClamp/autopatch_configuration.txt", "r") as json_infile:
                data = json.load(json_infile)
            self.pixel_size = data["pixel_size"]
            self.image_size = data["image_size"]
            self.pipette_orientation = data["pipette_orientation"]
            self.pipette_diameter = data["pipette_diameter"]
            self.rotation_angles = data["rotation_angles"]
            self.focus_offset = data["focus_offset"]
        except:
            self.write_constants_to_JSON()
        
    def write_constants_to_JSON(self):
        data = {
            "pixel_size": self.pixel_size,
            "image_size": self.image_size,
            "pipette_orientation": self.pipette_orientation,
            "pipette_diameter": self.pipette_diameter,
            "rotation_angles": self.rotation_angles,
            "focus_offset": self.focus_offset
            }
        with open("PatchClamp/autopatch_configuration.txt", "w") as json_outfile:
            json.dump(data, json_outfile)
    
    def _current_append_(self, values):
        """Append new values to a sliding window."""
        length = len(values)
        if self.n_c + length > self.window_size_c:
            # Buffer is full so make room.
            copySize = self.window_size_c - length
            self.current = self.current[-copySize:]
            self.n_c = copySize
        self.current = np.append(self.current, values)
        self.n_c += length
    
    def _voltage_append_(self, values):
        """Append new values to a sliding window."""
        length = len(values)
        if self.n_v + length > self.window_size_v:
            # Buffer is full so make room.
            copySize = self.window_size_v - length
            self.voltage = self.voltage[-copySize:]
            self.n_v = copySize
        self.voltage = np.append(self.voltage, values)
        self.n_v += length
    
    def _pressure_append_(self, values, timings):
        """Append new values to a sliding window."""
        length = 1
        if self.n_p + length > self.window_size_p:
            # Buffer is full so make room.
            copySize = self.window_size_p - length
            self.pressure = self.pressure[:,-copySize:]
            self.n_p = copySize
        self.pressure = np.append(self.pressure, np.array([[values],[timings]]), axis=1)
        self.n_p += length
    
    def _resistance_capacitance_append_(self, Rvalues, Cvalues):
        """Append new values to a sliding window."""
        length = 1
        if self.n_rc + length > self.window_size_rc:
            # Buffer is full so make room.
            copySize = self.window_size_rc - length
            self.resistance = self.resistance[-copySize:]
            self.capacitance = self.capacitance[-copySize:]
            self.n_rc = copySize
        self.resistance = np.append(self.resistance, Rvalues)
        self.capacitance = np.append(self.capacitance, Cvalues)
        self.n_rc += length
    
    
    @property
    def camerathread(self):
        return self._camerathread
    
    @camerathread.setter
    def camerathread(self, camerathread_handle):
        self._camerathread = camerathread_handle
        self._camerathread.start()
    
    @camerathread.deleter
    def camerathread(self):
        self._camerathread.stop()
        self._camerathread = None
    
    
    @property
    def sealtestthread(self):
        return self._sealtestthread
    
    @sealtestthread.setter
    def sealtestthread(self, sealtestthread_handle):
        logging.info('SealTestThread instantiated and wave set')
        self._sealtestthread = sealtestthread_handle
        self._sealtestthread.setWave(0.1, 0.01, 0)
        self._sealtestthread.start()
    
    @sealtestthread.deleter
    def sealtestthread(self):
        # self._sealtestthread.aboutToQuitHandler()
        self._sealtestthread.stop()
        self._sealtestthread = None
    
    
    @property
    def pressurethread(self):
        return self._pressurethread
    
    @pressurethread.setter
    def pressurethread(self, pressurecontroller):
        self._pressurethread = pressurecontroller
        self._pressurethread.parent = self
        self._pressurethread.start()
    
    @pressurethread.deleter
    def pressurethread(self):
        self._pressurethread.stop()
        self._pressurethread = None
        
        
    @property
    def micromanipulator(self):
        logging.info('micromanipulator get')
        return self._micromanipulator
    
    @micromanipulator.setter
    def micromanipulator(self, micromanipulator_handle):
        logging.info('micromanipulator set')
        self._micromanipulator = micromanipulator_handle
    
    @micromanipulator.deleter
    def micromanipulator(self):
        self._micromanipulator.stop()
        self._micromanipulator = None
    
    
    @property
    def objectivemotor(self):
        logging.info('objectivemotor get')
        return self._objectivemotor
    
    @objectivemotor.setter
    def objectivemotor(self, objective):
        logging.info('objectivemotor set')
        self._objectivemotor = objective
    
    @objectivemotor.deleter
    def objectivemotor(self):
        self._objectivemotor.disconnect()
        self._objectivemotor = None
    
    
    @property
    def XYstage(self):
        logging.info('XYstage get')
        return self._XYstage
    
    @XYstage.setter
    def XYstage(self, stage_handle):
        logging.info('XYstage set')
        self._XYstage = stage_handle
    
    @XYstage.deleter
    def XYstage(self):
        self._XYstage = None
    
    
    @property
    def pixel_size(self):
        return self._pixel_size
    
    @pixel_size.setter
    def pixel_size(self, size):
        logging.info('Set pixel size to: ' + str(size))
        if isinstance(size, float) or isinstance(size, int):
            self._pixel_size = size
        else:
            raise ValueError('pixelsize should be a float or integer')
    
    @pixel_size.deleter
    def pixel_size(self):
        self._pixel_size = None
    
    
    @property
    def image_size(self):
        return self._image_size
    
    @image_size.setter
    def image_size(self, size):
        width,height = size
        if type(width) and type(height) == float or int:
            logging.info('Set image size to: '+str(width)+'x'+str(height)+' pixels')
            self._image_size = [width, height]
        else:
            raise ValueError('Image size should have width and height of type float or integer')
    
    @image_size.deleter
    def image_size(self):
        self._image_size = [None, None]
    
    
    @property
    def pipette_orientation(self):
        return self._pipette_orientation
    
    @pipette_orientation.setter
    def pipette_orientation(self, angle):
        if isinstance(angle, float) or isinstance(angle, int):
            logging.info('Set pipette orientation: ' + str(angle) + ' degrees')
            self._pipette_orientation = angle
        else:
            raise ValueError('micromanipulator orientation should be a float or integer')
    
    @pipette_orientation.deleter
    def pipette_orientation(self):
        self._pipette_orientation = None
    
    
    @property
    def pipette_diameter(self):
        return self._pipette_diameter
    
    @pipette_diameter.setter
    def pipette_diameter(self, diameter):
        if isinstance(diameter, float) or isinstance(diameter, int):
            logging.info('Set pipette opening diameter: D = ' + str(diameter) + ' pixels')
            self._pipette_diameter = diameter
        else:
            raise ValueError('Pipette opening diameter should be a float or integer')
    
    @pipette_diameter.deleter
    def pipette_diameter(self):
        self._pipette_diameter = None
    
    
    @property
    def rotation_angles(self):
        return self._rotation_angles
    
    @rotation_angles.setter
    def rotation_angles(self, alphabetagamma):
        if len(alphabetagamma) == 3:
            alpha,beta,gamma = alphabetagamma
            if type(alpha) and type(beta) and type(gamma) == float or int:
                logging.info('Set rotation angles alpha beta gamma: '+str(alpha)+' '+str(beta)+' '+str(gamma))
                self._rotation_angles = [alpha, beta, gamma]
                self.R = (alpha, beta, gamma)
            else:
                raise ValueError('rotation angles should be integers or floats')
        else:
            raise ValueError('rotation angles should be a 3 element array or tuple')
    
    @rotation_angles.deleter
    def rotation_angles(self):
        self._rotation_angles = [0,0,0]
        del self.R
    
    @property
    def focus_offset(self):
        return self._focus_offset
    
    @focus_offset.setter
    def focus_offset(self, offset):
        self._focus_offset = offset
    
    @focus_offset.deleter
    def focus_offset(self):
        self._focus_offset = None
    
    
    @property
    def R(self):
        return self._R
    
    @R.setter
    def R(self, alphabetagamma):
        alpha,beta,gamma = alphabetagamma
        R_alpha = np.array([[1, 0, 0],
                            [0, np.cos(alpha), np.sin(alpha)],
                            [0, -np.sin(alpha), np.cos(alpha)]])
        R_beta = np.array([[np.cos(beta), 0, -np.sin(beta)],
                           [0, 1, 0],
                           [np.sin(beta), 0, np.cos(beta)]])
        R_gamma = np.array([[np.cos(gamma), np.sin(gamma), 0],
                            [-np.sin(gamma), np.cos(gamma), 0],
                            [0, 0, 1]])
        try:
            self._R = self._R @ R_gamma @ R_beta @ R_alpha
        except:
            self._R = R_gamma @ R_beta @ R_alpha
    
    @R.deleter
    def R(self):
        self._R = np.eye(3)
    
    
    @property
    def operation_mode(self):
        return self._operation_mode
    
    @operation_mode.setter
    def operation_mode(self, mode):
        if isinstance(mode, str):
            logging.info('Set operation mode for workers to: ' + mode)
            self._operation_mode = mode
        else:
            raise ValueError('operation mode for workers should be a string')
    
    @operation_mode.deleter
    def operation_mode(self):
        self._operation_mode = 'default'
    
    
    @property
    def pipette_coordinates_pair(self):
        return self._pipette_coordinates_pair
    
    @pipette_coordinates_pair.setter
    def pipette_coordinates_pair(self, coords):
        if isinstance(coords, np.ndarray):
            logging.info('Couple pipette coordinates:\n[X,Y,Z] = ' + str(coords[0,:]) + '\n[row,col,obj] = ' + str(coords[1,:]))
            if coords.shape == (2,3):
                self._pipette_coordinates_pair = coords
            else:
                raise ValueError('coordinates-pair size should be 2x3')
        else:
            raise ValueError('pipette coordinates should be a numpy.ndarray')
    
    @pipette_coordinates_pair.deleter
    def pipette_coordinates_pair(self):
        self._pipette_coordinates_pair = np.array([[None,None,None], [None,None,None]])
    
    
    @property
    def target_coordinates(self):
        return self._target_coordinates
    
    @target_coordinates.setter
    def target_coordinates(self, coords):
        logging.info('Set target coordinates:\n[row,col,obj] = ' + str(coords))
        if isinstance(coords, np.ndarray):
            if coords.shape == (3,):
                self._target_coordinates = coords
            else:
                raise ValueError('length of target coordinates must be 3')
        else:
            raise ValueError('target coordinates should be a numpy.ndarray')
    
    @target_coordinates.deleter
    def target_coordinates(self):
        self._pipette_coordinates = np.array([None, None, None])
    
    
    @property
    def resistance_reference(self):
        return self._resistance_reference
    
    @resistance_reference.setter
    def resistance_reference(self, resistance):
        logging.info('Pipette resistance reference set at: '+str(resistance*1e-6)+' MΩ')
        self._resistance_reference = resistance
    
    @resistance_reference.deleter
    def resistance_reference(self):
        self._resistance_reference = None
    
    
    @property
    def current(self):
        return self._current
    
    @current.setter
    def current(self, current_array):
        self._current = current_array
    
    @current.deleter
    def current(self):
        self._current = np.array([])
    
    @property
    def voltage(self):
        return self._voltage
    
    @voltage.setter
    def voltage(self, voltage_array):
        self._voltage = voltage_array
    
    @voltage.deleter
    def voltage(self):
        self._voltage = np.array([])
    
    @property
    def pressure(self):
        return self._pressure
    
    @pressure.setter
    def pressure(self, pressure_array):
        self._pressure = pressure_array
    
    @pressure.deleter
    def pressure(self):
        self._pressure = np.array([[],[]])
    
    @property
    def resistance(self):
        return self._resistance
    
    @resistance.setter
    def resistance(self, resistance_array):
        self._resistance = resistance_array
    
    @resistance.deleter
    def resistance(self):
        self._resistance = np.array([])
    
    @property
    def capacitance(self):
        return self._capacitance
    
    @capacitance.setter
    def capacitance(self, capacitance_array):
        self._capacitance = capacitance_array
    
    @capacitance.deleter
    def capacitance(self):
        self._capacitance = np.array([])
    
    
    @property
    def STOP(self):
        return self._STOP
    
    @STOP.setter
    def STOP(self, state):
        """
        # 1. Stop QThreads
        # 2. Stop hardware from moving
        # 3. Set pressure to ATM
        # 4. Set current/voltage to 0
        """
        if state == True:
            self._STOP = True
            logging.info('Emergency stop active')
            self.worker.STOP = True
        elif state == False:
            self._STOP = False
            logging.info('Emergency stop standby')
            self.worker.STOP = False
        else:
            raise ValueError('Emergency stop should be either TRUE or FALSE')
    