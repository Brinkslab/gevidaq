# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 15:14:54 2020

@author: xinmeng
"""

import os
import time
from ctypes import *


class KCube:
    """
    Thorlabs' KBD101 K-Cube Brushless DC Motor Controller provides local and
    computerized control of a single motor axis. It features a top-mounted control
    panel with a velocity wheel that supports four-speed bidirectional control with
    forward and reverse jogging as well as position presets. The digital display on
    the top panel includes a backlight that can be dimmed or turned off using the
    top panel menu options. The front of the unit contains two bidirectional trigger
    ports that can be used to read a 5 V external logic signal or output a 5 V logic
    signal to control external equipment. Each port can be independently configured
    to control the logic level or to set the trigger as an input or output.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set up serial number variable
        self.serialNumber = c_char_p("28251139".encode("utf-8"))

        self.moveTimeout = 60.0
        self.messageType = c_ushort()
        self.messageID = c_ushort()
        self.messageData = c_ulong()

    def initialize(self):
        # Load the dll file.
        self.lib = cdll.LoadLibrary(
            r"C:\Labsoftware\Thorlabs\Thorlabs.MotionControl.KCube.BrushlessMotor.dll"
        )

        # Build device list
        self.lib.TLI_BuildDeviceList()

        # Set up device
        self.lib.BMC_Open(self.serialNumber)
        self.lib.BMC_StartPolling(self.serialNumber, c_int(200))
        self.lib.BMC_EnableChannel(self.serialNumber)

        time.sleep(3)

        self.lib.BMC_ClearMessageQueue(self.serialNumber)

    def Home(self):
        # Home device
        print("Homing Device")
        homeStartTime = time.time()
        self.lib.BMC_Home(self.serialNumber)

        homed = False

        while homed == False:
            self.lib.BMC_GetNextMessage(
                self.serialNumber,
                byref(self.messageType),
                byref(self.messageID),
                byref(self.messageData),
            )
            if (self.messageID.value == 0 and self.messageType.value == 2) or (
                time.time() - homeStartTime
            ) > self.moveTimeout:
                homed = True
        self.lib.BMC_ClearMessageQueue(self.serialNumber)

    def Move(self, pos):
        # Move to absolue position, in mm.
        deviceUnit = c_int()

        # here, we move to position in real units (mm)
        realUnit = c_double(pos)

        # Load settings for attached stage
        self.lib.BMC_LoadSettings(self.serialNumber)

        # Convert real units to device units
        self.lib.BMC_GetDeviceUnitFromRealValue(
            self.serialNumber, realUnit, byref(deviceUnit), 0
        )

        # Send move command
        print("Moving Device")

        moveStartTime = time.time()
        self.lib.BMC_MoveToPosition(self.serialNumber, deviceUnit)

        moved = False

        while moved == False:
            self.lib.BMC_GetNextMessage(
                self.serialNumber,
                byref(self.messageType),
                byref(self.messageID),
                byref(self.messageData),
            )

            if (self.messageID.value == 1 and self.messageType.value == 2) or (
                time.time() - moveStartTime
            ) > self.moveTimeout:
                moved = True

    def Exit(self):
        # Clean up and exit
        self.lib.BMC_ClearMessageQueue(self.serialNumber)

        self.lib.BMC_StopPolling(self.serialNumber)

        self.lib.BMC_Close(self.serialNumber)

        print("K-cube connection closed.")


if __name__ == "__main__":
    motor = KCube()
    motor.initialize()
    motor.Home()
    motor.Move(32)

    time.sleep(1)

    motor.Exit()
