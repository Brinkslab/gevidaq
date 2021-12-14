# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 09:49:25 2021

@author: tvdrb
"""

import serial


class PressureController(serial.Serial):
    """ Pressure Controller communication commands for serial communication.
    This class is for controlling the Pressure Controller.
    """
    
    def __init__(self, address, baud):
        super().__init__(port=address, baudrate=baud, timeout=1)
        self.ENDOFLINE = '\n'   # Carriage return
    
    
    def readFlush(self):
        # Read serial port and flush input
        if self.inWaiting():
            response = self.read_until(self.ENDOFLINE.encode('ascii'))
            while self.inWaiting():
                response = self.read_until(self.ENDOFLINE.encode('ascii'))
            try:
                response = response.decode('utf-8')
            except:
                response = ""
        else:
            response = ""
        
        return response
    
    
    def setPres(self, pressure):
        """
        This sets the pressure to the given target pressure value.
            Send: P 100
        """
        command = "P %d" % pressure + self.ENDOFLINE
        
        # Encode the command to ascii and send to the device
        self.write(command.encode('ascii'))
    
    
    def doPulse(self, magnitude):
        """
        This gives a pressure pulse with the desired magnitude.
            Send: S -100
        """
        command = "S %d" % magnitude + self.ENDOFLINE
        
        # Encode the command to ascii and send to the device
        self.write(command.encode('ascii'))
    
    
    def goIdle(self):
        """
        This lets the pressure controller enter idle mode. The device is does
        not turn off! It only turns off the lcd, the pumps, and the valves
        while waiting for a 
        """
        command = "IDLE" + self.ENDOFLINE
        
        # Encode the command to ascii and send to the device
        self.write(command.encode('ascii'))
    
    
    def wakeUp(self):
        """
        This wakes up the pressure controller from idle mode. Waking up resets
        the pressure controller, it includes recalibration of the pressure
        sensors.
        """
        command = "W8KE" + self.ENDOFLINE
        
        # Encode the command to ascii and send to the device
        self.write(command.encode('ascii'))


if __name__ == "__main__":
    PS = PressureController(address='COM21', baud=9600)
    
    
    