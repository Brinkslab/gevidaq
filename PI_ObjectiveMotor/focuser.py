# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:41:41 2019

@author: xinmeng
"""
import sys
import os
import time

if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname + "/../")

try:
    from pipython import GCSDevice
    from pipython import pitools
except:
    print("pipython not configured.")


class PIMotor:
    def __init__(self):
        """
        Provide a device, connected via the PI GCS DLL.

        def pipython.gcsdevice.GCSDevice.__init__	(	 	self,
                devname = '',
                gcsdll = '')

        Parameters
        devname	: Name of device, chooses according DLL which defaults to PI_GCS2_DLL.
        gcsdll	: Name or path to GCS DLL to use, overwrites 'devname'.

        Returns
        -------
        None.

        """
        try:
            CONTROLLERNAME = "C-863.11"
            # STAGES = None
            STAGES = "M-110.1DG"
            REFMODE = "FNL"

            # Get the path to dll in the same folder.
            abspath = os.path.abspath(__file__)
            dname = os.path.dirname(abspath) + "/PI_GCS2_DLL_x64.dll"
            print(dname)

            self.pidevice = GCSDevice(gcsdll=dname)
            print(self.pidevice.EnumerateUSB())
            # InterfaceSetupDlg() is an interactive dialog. There are other methods to
            # connect to an interface without user interaction.
            serialstring = self.pidevice.EnumerateUSB()
            print(serialstring[0])
            # pidevice.InterfaceSetupDlg(key='sample')
            # pidevice.ConnectRS232(comport=1, baudrate=115200)
            self.pidevice.ConnectUSB(serialnum="PI C-863 Mercury SN 0185500828")
            # pidevice.ConnectTCPIP(ipaddress='192.168.178.42')

            # Each PI controller supports the qIDN() command which returns an
            # identification string with a trailing line feed character which
            # we "strip" away.

            print("connected: {}".format(self.pidevice.qIDN().strip()))

            # Show the version info which is helpful for PI support when there
            # are any issues.

            if self.pidevice.HasqVER():
                print("version info: {}".format(self.pidevice.qVER().strip()))
        #    allaxes = self.pidevice.qSAI_ALL()
        #        return self.pidevice
        except:
            print("PI device not initilized.")

    def move(self, target_pos):
        # pidevice.StopAll()
        # pidevice.SVO(pidevice.axes, [True] * len(pidevice.axes))
        # pitools.waitontarget(pidevice, axes=pidevice.axes)

        #            pitools.startup(pidevice, stages=STAGES, refmode=REFMODE)

        # Now we query the allowed motion range of all connected stages.
        # GCS commands often return an (ordered) dictionary with axes/channels
        # as "keys" and the according values as "values".

        # rangemin = list(pidevice.qTMN().values())
        # rangemax = list(pidevice.qTMX().values())
        # ranges = zip(rangemin, rangemax)

        targets = [target_pos]
        self.pidevice.MOV(self.pidevice.axes, targets)
        pitools.waitontarget(self.pidevice)
        time.sleep(0.3)
        positions = self.pidevice.qPOS(self.pidevice.axes)
        for axis in self.pidevice.axes:
            print("position of axis {} = {:.5f}".format(axis, positions[axis]))

    def GetCurrentPos(self):
        # positions is a dictionary with key being axis name, here '1'.
        positions = self.pidevice.qPOS(self.pidevice.axes)

        return positions["1"]

    def CloseMotorConnection(self):
        self.pidevice.CloseConnection()


if __name__ == "__main__":
    # To see what is going on in the background you can remove the following
    # two hashtags. Then debug messages are shown. This can be helpful if
    # there are any issues.

    # import logging
    # logging.basicConfig(level=logging.DEBUG)
    pi = PIMotor()
    #    pi_device = pi.ConnectPIMotor

    #    PIMotor.move(pi.pidevice, 3.455)
    print(pi.GetCurrentPos())
    pi.CloseMotorConnection()
