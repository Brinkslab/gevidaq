#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 09:21:44 2020

@author: Izak de Heer

2021-1 
Modified based on Izak's code for higher level interface.
"""

from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname + "/../")


def _transform_x(p, *kwargs):
    order = int(np.sqrt(len(kwargs)))
    c = np.reshape(np.asarray(kwargs), (order, order))
    return np.polynomial.polynomial.polyval2d(p[:, 0], p[:, 1], c)


def _transform_y(p, *kwargs):
    order = int(np.sqrt(len(kwargs)))
    c = np.reshape(np.asarray(kwargs), (order, order))
    return np.polynomial.polynomial.polyval2d(p[:, 0], p[:, 1], c)


def polynomial2DFit(p, q, order=1):
    """
    Input an array of coordinates p and q in frame P and Q. The function returns
    a function matrix that transforms P into Q using polynomial transform.

    param p, q = np.array([p1_x, p1_y],
                          [p2_x, p2_y,
                            .., ..])

    param order = order of polynomial transform, order should be smaller than
                  the number of input points

    return = stacked polynomial coefficients matrices, rounded to 4 decimals

    """
    if isinstance(p, list):
        p = np.array(p)
    if isinstance(q, list):
        q = np.array(q)

    init_param = np.ones((order + 1, order + 1))
    params_x, _ = optimize.curve_fit(_transform_x, p, q[:, 0], init_param)
    params_y, _ = optimize.curve_fit(_transform_y, p, q[:, 1], init_param)

    return np.stack(
        (
            np.reshape(params_x, (order + 1, order + 1)),
            np.reshape(params_y, (order + 1, order + 1)),
        ),
        axis=2,
    )


def transform(coord, c):
    """
    param coord: np.ndarray of shape nx2 containing x and y coordinates.
    param c: matrix containing the transformation coefficients.

    """
    if isinstance(coord, list):
        coord = np.array(coord)

    if coord.ndim == 1:
        new_coord_x = np.polynomial.polynomial.polyval2d(coord[0], coord[1], c[:, :, 0])
        new_coord_y = np.polynomial.polynomial.polyval2d(coord[0], coord[1], c[:, :, 1])
    else:
        new_coord_x = np.polynomial.polynomial.polyval2d(
            coord[:, 0], coord[:, 1], c[:, :, 0]
        )
        new_coord_y = np.polynomial.polynomial.polyval2d(
            coord[:, 0], coord[:, 1], c[:, :, 1]
        )
    return np.transpose(np.stack((new_coord_x, new_coord_y)))


# =============================================================================
# Integrated transformation
#
# --Based on Izak's work, warp it in easy to use way.
# =============================================================================


def general_coordinates_transformation(
    coordinates_list, direction="Camera2Galvo", **kwargs
):
    """
    Transform input list of coordinates, return transformed corresponding coordinates.

    Parameters
    ----------
    camera_coordinates_list : list or np.array
        List of camera coordinates np.array, i.e.,
        [np.array([row, col]), np.array([row, col])], in 2048*2048 size.

    Returns
    -------
    Transformed_coordinates : np.array
        np.array of n rows 2 column.

    """

    if direction == "Camera2Galvo":
        # ===============================================================
        # Transform from camera pixel positions to galvo voltage values.
        # ===============================================================
        # Load transformation
        transform_matrix_cam2galvo = load_transformation("galvo_transformation")

        Transformed_coordinates = transform_coordinates(
            coordinates_list, transform_matrix_cam2galvo
        )

    elif direction == "Galvo2Camera":
        # ===============================================================
        # Transform from galvo voltage values to camera pixel positions.
        # ===============================================================
        # Load transformation
        transform_matrix_cam2galvo = load_transformation("galvo_transformation")

        # To reverse and get the transform matrix from galvo to camera,
        # first need to using existing transform to get coordinates of both sides,
        # then transform polynomial2DFit the reverse way.
        camera_array_reverse_purpose = [
            [1800.0, 1800.0],
            [1800.0, 1200.0],
            [1800.0, 600.0],
            [1200.0, 1800.0],
            [1200.0, 1200.0],
            [1200.0, 600.0],
            [600.0, 1800.0],
            [600.0, 1200.0],
            [600.0, 600.0],
        ]

        galvo_coordinates_reverse_purpose = transform_coordinates(
            camera_array_reverse_purpose, transform_matrix_cam2galvo
        )

        # Fit to get the transformation
        transform_matrix_galvo2cam = polynomial2DFit(
            galvo_coordinates_reverse_purpose, camera_array_reverse_purpose, order=1
        )

        Transformed_coordinates = transform_coordinates(
            coordinates_list, transform_matrix_galvo2cam
        )

    elif direction == "Camera2PMT":
        # ===============================================================
        # Transform from camera pixel positions to PMT image pixel positions.
        # ===============================================================
        # Load transformation
        transform_matrix_cam2galvo = load_transformation("galvo_transformation")
        # Transform to galvo voltages
        Transformed_coordinates_voltage = transform_coordinates(
            coordinates_list, transform_matrix_cam2galvo
        )
        # Transform voltages to PMT image pixel positions
        Transformed_coordinates = transform_between_PMT_Galvo(
            Transformed_coordinates_voltage, "Galvo2PMT", kwargs["scanning_config"]
        )

    elif direction == "PMT2Camera":
        # ===============================================================
        # Transform from camera pixel positions to PMT image pixel positions.
        # ===============================================================

        # Transform PMT image pixel position to voltages
        galvo_voltage_coordinates = transform_between_PMT_Galvo(
            coordinates_list, "PMT2Galvo", kwargs["scanning_config"]
        )
        # Load transformation
        transform_matrix_cam2galvo = load_transformation("galvo_transformation")

        # To reverse and get the transform matrix from galvo to camera,
        # first need to using existing transform to get coordinates of both sides,
        # then transform polynomial2DFit the reverse way.
        camera_array_reverse_purpose = [
            [1800.0, 1800.0],
            [1800.0, 1200.0],
            [1800.0, 600.0],
            [1200.0, 1800.0],
            [1200.0, 1200.0],
            [1200.0, 600.0],
            [600.0, 1800.0],
            [600.0, 1200.0],
            [600.0, 600.0],
        ]

        galvo_coordinates_reverse_purpose = transform_coordinates(
            camera_array_reverse_purpose, transform_matrix_cam2galvo
        )

        # Fit to get the transformation
        transform_matrix_galvo2cam = polynomial2DFit(
            galvo_coordinates_reverse_purpose, camera_array_reverse_purpose, order=1
        )

        Transformed_coordinates = transform_coordinates(
            galvo_voltage_coordinates, transform_matrix_galvo2cam
        )

    # Set to int.
    Transformed_coordinates = Transformed_coordinates.astype(int)

    return Transformed_coordinates


def load_transformation(target):
    """
    The transformation direction here is ALL STARTING FROM CAMERA COORDINATES!

    Parameters
    ----------
    target : string
        Name of the transformation file.

    Returns
    -------
    transform_matrix : np array
        transform_matrix.

    """
    # Load transformation
    transformation_filename = os.path.join(
        os.getcwd() + "\CoordinatesManager\Registration\{}".format(target)
    )
    transform_matrix_flat = np.loadtxt(transformation_filename)
    # Reshape the text matrix into (2, 2, 2)
    transform_matrix = np.reshape(
        transform_matrix_flat, (transform_matrix_flat.shape[1], -1, 2)
    )

    return transform_matrix


def transform_coordinates(list_of_coordinates, transform_matrix):
    """
    Given list of roi positions in camera image, transform into corrseponding
    voltage positions.

    Parameters
    ----------
    list_of_coordinates : list
        List of np.array.

    Returns
    -------
    new_list_of_coordinates : np array
        List of np.array.

    """
    new_list_of_coordinates = []
    for roi in list_of_coordinates:
        new_list_of_coordinates.append(transform(roi, transform_matrix))

    return np.array(new_list_of_coordinates)


def transform_between_PMT_Galvo(coordinates_list, direction, scanning_config):
    """
    Transform between PMT image pixel positions and galvo voltage values.

    Parameters
    ----------
    coordinates_list : list or np.array
        DESCRIPTION.
    direction : string
        Direction of transformation.
    scanning_config : list
        First element being scanning voltage, second being pixel number of PMT image.

    Returns
    -------
    Transformed_coordinates : np.array
        DESCRIPTION.

    """
    galvo_scanning_voltage_range = scanning_config[0]
    PMT_image_pixel_number = scanning_config[1]

    Transformed_coordinates = []
    for each_coordinate in coordinates_list:
        # If is a list,  convert to np array
        if isinstance(each_coordinate, list):
            each_coordinate = np.array(each_coordinate)

        if direction == "Galvo2PMT":
            Transformed_coordinates.append(
                (
                    (each_coordinate + galvo_scanning_voltage_range)
                    / (galvo_scanning_voltage_range * 2)
                    * PMT_image_pixel_number
                ).astype(int)
            )

        elif direction == "PMT2Galvo":
            Transformed_coordinates.append(
                (
                    (each_coordinate / PMT_image_pixel_number)
                    * galvo_scanning_voltage_range
                    * 2
                    - galvo_scanning_voltage_range
                ).astype(int)
            )

    return np.array(Transformed_coordinates)


if __name__ == "__main__":
    new_list_of_coordinates = general_coordinates_transformation(
        np.array([[251, 249], [100, 100]]), "PMT2Camera", scanning_config=[5, 500]
    )
    print(new_list_of_coordinates[0])
    # grid_points_x = 2
    # grid_points_y = 3

    # x_coords = np.linspace(0, 768, grid_points_x+2)[1:-1].astype(int)
    # y_coords = np.linspace(0, 1024, grid_points_y+2)[1:-1].astype(int)

    # px = np.linspace(0, 768, 4)[1:-1].astype(int)
    # py = np.linspace(0, 1024, 5)[1:-1].astype(int)

    # qx = np.array((571, 1121, 1730, 648, 1120, 1648))
    # qy = np.array((807, 797, 861, 1411, 1348, 1340))

    # coord_out = np.reshape((np.meshgrid(x_coords, y_coords)), (2, -1), order='F').transpose()
    # coord_in = np.stack((qy,qx), axis=1)

    # soln1 = np.round(polynomial2DFit(coord_in, coord_out), 3)

    # print('in x:')
    # print(soln1[:,:,0])
    # print('in y:')
    # print(soln1[:,:,1])
