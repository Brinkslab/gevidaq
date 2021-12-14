# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:54:51 2020

@author: xinmeng
"""

# =============================================================================
# -------------------------Auto focus workflow---------------------------------
# --1. Find local minima range. (one direction until value getting smaller)
# --2. Bisection to find the optimal.
# =============================================================================

import os

# Ensure that the Widget can be run either independently or as part of Tupolev.
if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname + "/../")

from PI_ObjectiveMotor.focuser import PIMotor
from ImageAnalysis.ImageProcessing import ProcessImage
from GalvoWidget.GalvoScan_backend import RasterScan
from HamamatsuCam.HamamatsuActuator import CamActuator
from NIDAQ.DAQoperator import DAQmission
import skimage.external.tifffile as skimtiff
import time
import matplotlib.pyplot as plt
import numpy as np


class FocusFinder:
    def __init__(
        self,
        source_of_image="PMT",
        init_search_range=0.010,
        total_step_number=5,
        imaging_conditions={"edge_volt": 5},
        motor_handle=None,
        camera_handle=None,
        twophoton_handle=None,
        *args,
        **kwargs
    ):
        """


        Parameters
        ----------
        source_of_image : string, optional
            The input source of image. The default is PMT.
        init_search_range : int, optional
            The step size when first doing coarse searching. The default is 0.010.
        total_step_number : int, optional
            Number of steps in total to find optimal focus. The default is 5.
        imaging_conditions : list
            Parameters for imaging.
            For PMT, it specifies the scanning voltage.
            For camera, it specifies the AOTF voltage and exposure time.
        motor_handle : TYPE, optional
            Handle to control PI motor. The default is None.
        twophoton_handle : TYPE, optional
            Handle to control Insight X3. The default is None.

        Returns
        -------
        None.

        """
        super().__init__(*args, **kwargs)

        # The step size when first doing coarse searching.
        self.init_search_range = init_search_range

        # Number of steps in total to find optimal focus.
        self.total_step_number = total_step_number

        # Parameters for imaging.
        self.imaging_conditions = imaging_conditions

        if motor_handle == None:
            # Connect the objective if the handle is not provided.
            self.pi_device_instance = PIMotor()
        else:
            self.pi_device_instance = motor_handle

        # Current position of the focus.
        self.current_pos = self.pi_device_instance.GetCurrentPos()

        # Number of steps already tried.
        self.steps_taken = 0
        # The focus degree of previous position.
        self.previous_degree_of_focus = 0
        # Number of going backwards.
        self.turning_point = 0
        # The input source of image.
        self.source_of_image = source_of_image
        if source_of_image == "PMT":
            self.galvo = RasterScan(
                Daq_sample_rate=500000, edge_volt=self.imaging_conditions["edge_volt"]
            )
        elif source_of_image == "Camera":
            if camera_handle == None:
                # If no camera instance fed in, initialize camera.
                self.HamamatsuCam_ins = CamActuator()
                self.HamamatsuCam_ins.initializeCamera()
            else:
                self.HamamatsuCam_ins = camera_handle

    def gaussian_fit(self, move_to_focus=True):

        # The upper edge.
        upper_position = self.current_pos + self.init_search_range
        # The lower edge.
        lower_position = self.current_pos - self.init_search_range

        # Generate the sampling positions.
        sample_positions = np.linspace(
            lower_position, upper_position, self.total_step_number
        )

        degree_of_focus_list = []
        for each_pos in sample_positions:
            # Go through each position and write down the focus degree.
            degree_of_focus = self.evaluate_focus(round(each_pos, 6))
            degree_of_focus_list.append(degree_of_focus)
        print(degree_of_focus_list)

        # try:
        #     interpolated_fitted_curve = ProcessImage.gaussian_fit(degree_of_focus_list)

        #     # Generate the inpterpolated new focus position axis.
        #     x_axis_new = np.linspace(
        #         lower_position, upper_position, len(interpolated_fitted_curve)
        #     )

        #     # Generate a dictionary and find the position where has the highest focus degree.
        #     max_focus_pos = dict(zip(interpolated_fitted_curve, x_axis_new))[
        #         np.amax(interpolated_fitted_curve)
        #     ]

        #     if True:  # Plot the fitting.
        #         plt.plot(
        #             sample_positions,
        #             np.asarray(degree_of_focus_list),
        #             "b+:",
        #             label="data",
        #         )
        #         plt.plot(x_axis_new, interpolated_fitted_curve, "ro:", label="fit")
        #         plt.legend()
        #         plt.title("Fig. Fit for focus degree")
        #         plt.xlabel("Position")
        #         plt.ylabel("Focus degree")
        #         plt.show()

        #     max_focus_pos = round(max_focus_pos, 6)
        #     print(max_focus_pos)
        #     self.pi_device_instance.move(max_focus_pos)
        #     # max_focus_pos_focus_degree = self.evaluate_focus(round(max_focus_pos, 6))
        # except:
        print("Fitting failed. Find max in the list.")

        max_focus_pos = sample_positions[
            degree_of_focus_list.index(max(degree_of_focus_list))
        ]
        print(max_focus_pos)

        if move_to_focus == True:
            self.pi_device_instance.move(max_focus_pos)

        return max_focus_pos

    def bisection(self):
        """
        Bisection way of finding focus.

        Returns
        -------
        mid_position : float
            DESCRIPTION.

        """
        # The upper edge in which we run bisection.
        upper_position = self.current_pos + self.init_search_range
        # The lower edge in which we run bisection.
        lower_position = self.current_pos - self.init_search_range

        for step_index in range(1, self.total_step_number + 1):
            # In each step of bisection finding.

            # In the first round, get degree of focus at three positions.
            if step_index == 1:
                # Get degree of focus in the mid.
                mid_position = (upper_position + lower_position) / 2
                degree_of_focus_mid = self.evaluate_focus(mid_position)
                print("mid focus degree: {}".format(round(degree_of_focus_mid, 5)))

                # Break the loop if focus degree is below threshold which means
                # that there's no cell in image.
                if not ProcessImage.if_theres_cell(self.galvo_image.astype("float32")):
                    print("no cell")
                    mid_position = False
                    break

                # Move to top and evaluate.
                degree_of_focus_up = self.evaluate_focus(obj_position=upper_position)
                print("top focus degree: {}".format(round(degree_of_focus_up, 5)))
                # Move to bottom and evaluate.
                degree_of_focus_low = self.evaluate_focus(obj_position=lower_position)
                print("bot focus degree: {}".format(round(degree_of_focus_low, 5)))
                # Sorting dicitonary of degrees in ascending.
                biesection_range_dic = {
                    "top": [upper_position, degree_of_focus_up],
                    "bot": [lower_position, degree_of_focus_low],
                }

            # In the next rounds, only need to go to center and update boundaries.
            elif step_index > 1:
                # The upper edge in which we run bisection.
                upper_position = biesection_range_dic["top"][0]
                # The lower edge in which we run bisection.
                lower_position = biesection_range_dic["bot"][0]

                # Get degree of focus in the mid.
                mid_position = (upper_position + lower_position) / 2
                degree_of_focus_mid = self.evaluate_focus(mid_position)

                print("Current focus degree: {}".format(round(degree_of_focus_mid, 5)))

            # If sits in upper half, make the middle values new bottom.
            if biesection_range_dic["top"][1] > biesection_range_dic["bot"][1]:
                biesection_range_dic["bot"] = [mid_position, degree_of_focus_mid]
            else:
                biesection_range_dic["top"] = [mid_position, degree_of_focus_mid]

            print(
                "The upper pos: {}; The lower: {}".format(
                    biesection_range_dic["top"][0], biesection_range_dic["bot"][0]
                )
            )

        return mid_position

    def evaluate_focus(self, obj_position=None):
        """
        Evaluate the focus degree of certain objective position.

        Parameters
        ----------
        obj_position : float, optional
            The target objective position. The default is None.

        Returns
        -------
        degree_of_focus : float
            Degree of focus.

        """

        if obj_position != None:
            self.pi_device_instance.move(obj_position)

        # Get the image.
        if self.source_of_image == "PMT":
            self.galvo_image = self.galvo.run()
            plt.figure()
            plt.imshow(self.galvo_image)
            plt.show()

            if False:
                with skimtiff.TiffWriter(
                    os.path.join(
                        r"M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Xin\2020-11-17 gaussian fit auto-focus cells\trial_11",
                        str(obj_position).replace(".", "_") + ".tif",
                    )
                ) as tif:
                    tif.save(self.galvo_image.astype("float32"), compress=0)

            degree_of_focus = ProcessImage.local_entropy(
                self.galvo_image.astype("float32")
            )

        elif self.source_of_image == "Camera":
            # First configure the AOTF.
            self.AOTF_runner = DAQmission()
            # Find the AOTF channel key
            for key in self.imaging_conditions:
                if "AO" in key:
                    # like '488AO'
                    AOTF_channel_key = key

            # Set the AOTF first.
            self.AOTF_runner.sendSingleDigital("blankingall", True)
            self.AOTF_runner.sendSingleAnalog(
                AOTF_channel_key, self.imaging_conditions[AOTF_channel_key]
            )

            # Snap an image from camera
            self.camera_image = self.HamamatsuCam_ins.SnapImage(
                self.imaging_conditions["exposure_time"]
            )
            time.sleep(0.5)

            # Set back AOTF
            self.AOTF_runner.sendSingleDigital("blankingall", False)
            self.AOTF_runner.sendSingleAnalog(AOTF_channel_key, 0)

            plt.figure()
            plt.imshow(self.camera_image)
            plt.show()

            if False:
                with skimtiff.TiffWriter(
                    os.path.join(
                        r"M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\Xin\2021-03-06 Camera AF\beads",
                        str(obj_position).replace(".", "_") + ".tif",
                    )
                ) as tif:
                    tif.save(self.camera_image.astype("float32"), compress=0)

            degree_of_focus = ProcessImage.variance_of_laplacian(
                self.camera_image.astype("float32")
            )

        time.sleep(0.2)

        return degree_of_focus

    # def explore(self, image):

    #     # Calculate the focus degree.
    #     self.current_degree_of_focus = ProcessImage.variance_of_laplacian(image)

    #     # If a turning point is met (except first attempt at wrong direction),
    #     # starts to bisection.
    #     if self.turning_point >= 1 and self.steps_taken != 1:
    #         self.steps_taken_after_turning =

    #     self.init_search_range *= (1/2) ** (self.steps_taken_after_turning)

    #     # if focus degree increases, move one step forwards.
    #     if self.current_degree_of_focus > self.previous_degree_of_focus:
    #         PIMotor.move(self.pi_device_instance.pidevice, self.current_pos + self.init_search_range)

    #     else: # else move downwards.

    #         self.init_search_range *= -1

    #         # If the first attempt goes towards the wrong direction,
    #         # turn around and move one step the other way.
    #         if self.steps_taken == 1 and self.turning_point == 1:
    #             # Move two step downwards.
    #             PIMotor.move(self.pi_device_instance.pidevice, self.current_pos + 2 * self.init_search_range)
    #             # Clean up trace, make sure the correct condition for first attempt.
    #             self.steps_taken = 0
    #             self.turning_point = 0

    #         else: # In normal bisection situation.
    #             PIMotor.move(self.pi_device_instance.pidevice, self.current_pos + self.init_search_range)
    #             # Add one turning point.
    #             self.turning_point += 1

    #     self.current_pos = self.pi_device_instance.GetCurrentPos()
    #     # Cast the current focus degree to previous for next round.
    #     self.previous_degree_of_focus = self.current_degree_of_focus

    #     # Update total number of steps.
    #     self.steps_taken += 1


if __name__ == "__main__":
    # ins = FocusFinder()
    # ins.total_step_number = 7
    # ins.init_search_range = 0.013
    # ins.gaussian_fit() # will return false if there's no cell in view.
    # ins.pi_device_instance.CloseMotorConnection()

    ins = FocusFinder(
        source_of_image="Camera",
        imaging_conditions={"488AO": 3, "exposure_time": 0.005},
    )
    ins.total_step_number = 7
    ins.init_search_range = 0.013
    ins.gaussian_fit()  # will return false if there's no cell in view.
    ins.pi_device_instance.CloseMotorConnection()
