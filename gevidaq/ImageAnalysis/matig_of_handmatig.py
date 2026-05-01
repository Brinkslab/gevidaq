import csv
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector
from skimage.io import imread

"""
Author: Ian Bot
Date: 28/3/2024
Description: This script is used to speed up the selection process of a box for the background subtraction.

NOTE: ONLY USE Q TO GO TO NEXT IMAGE, DO NOT USE THE CLOSE BUTTON ON THE PLOT WINDOW
"""


def search_for_tif_files(directory: str) -> list:
    """
    This function searches a directory for every .tif file and returns the full directory path
    :param directory: str: the full directory path to search for .tif files
    :return tif_files: list: list of full directory paths to .tif files in the main directory:
    :return tif_files_path: list: list of .tif files in the main directory
    """
    tif_files: list = []
    tif_files_path: list = []
    folder_path: list = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("fps.TIF") or file.endswith("fps.tif"):
                folder_path.append(root)
                tif_files_path.append(os.path.join(root, file))
                x = "data\\"
                name = f"{root.split(x)[1]} {file[:-4]}"
                tif_files.append(name)

    tif_files = [x.replace("\\", " ") for x in tif_files]
    return tif_files, tif_files_path, folder_path


def save_background(
    videostack_max: np.ndarray,
    save_loc_background: str,
    x: int,
    y: int,
    box_size: int = 30,
) -> None:
    """
    This function saves the background image with the selected box drawn on it. Helpful to double check if the selected box contains the cell
    :param videostack_max: np.ndarray: the maximum projection of the videostack to plot the box over
    :param save_loc_background: str: the full directory path to save the image including name and extension
    :param x: int: left x coordinate of the box
    :param y: int: top y coordinate of the box
    :param box_size: int: the size of the box to draw
    :return: None
    """

    plt.imshow(videostack_max)
    plt.plot([x, x + box_size], [y, y], "r")
    plt.plot([x, x + box_size], [y + box_size, y + box_size], "r")
    plt.plot([x, x], [y, y + box_size], "r")
    plt.plot([x + box_size, x + box_size], [y, y + box_size], "r")
    plt.colorbar()

    plt.savefig(save_loc_background, dpi=1000)
    plt.close()


def select_box(path: str, box_size: int = 30) -> tuple:
    """
    This function allows the user to select a box in the image to use as background.
    Lowest value for x and y are used and a box of size box_size is drawn. This equates to the upper left corner
    :param path: str: the full directory path to the image
    :param box_size: int: the size of the box to draw
    :return: tuple: the x and y coordinates of the upper left corner of the box
    """
    import matplotlib

    matplotlib.use("Qt5Agg")

    # function 'loaned' from matplotlib documentation (please don't ask how it works)
    def line_select_callback(eclick, erelease):
        "eclick and erelease are the press and release events"
        global x1, y1, x2, y2
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        x1 = np.round(x1)
        y1 = np.round(y1)
        x2 = np.round(x2)
        y2 = np.round(y2)

        logging.info("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))

        def prRed(skk):
            logging.info("\033[91m {}\033[00m".format(skk))

        if np.abs(x1 - x2) < box_size:
            prRed("x selection too small")
        if np.abs(y1 - y2) < box_size:
            prRed("y selection too small")

    # function 'loaned' from matplotlib documentation (please don't ask how it works)
    def toggle_selector(event):
        if event.key in ["Q", "q"] and toggle_selector.RS.active:
            logging.info("going to next image")
            toggle_selector.RS.set_active(False)

    videostack = imread(path)
    videostack_max = np.max(videostack, axis=0)
    fig, current_ax = plt.subplots()
    plt.imshow(videostack_max)

    # code 'loaned' from matplotlib documentation
    logging.info("\n      click  -->  release")
    toggle_selector.RS = RectangleSelector(
        current_ax,
        line_select_callback,
        useblit=True,
        button=[1, 3],  # don't use middle button
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
    )
    plt.connect("key_press_event", toggle_selector)
    plt.show()
    # end of 'loaned' code

    save_loc = os.path.join(path[:-11], "background selection.png")
    save_background(videostack_max, save_loc, min(x1, x2), min(y1, y2))

    return min(x1, x2), min(y1, y2)


def main(directory):
    # NOTE: ONLY USE Q TO GO TO NEXT IMAGE, DO NOT USE THE CLOSE BUTTON ON THE PLOT WINDOW
    # MARCO doe hier je directory: algemene data\\ directory

    logging.info(" starting...")

    if not os.path.isfile(f"{directory}cell_locations.csv"):
        with open(
            f"{directory}cell_locations.csv", mode="w", newline=""
        ) as file:
            logging.info(
                f" creating csv file... \n file location: {directory}cell_locations.csv \n"
            )
            writer = csv.writer(file)
            writer.writerow(["Cell", "X", "Y"])
    else:
        logging.info(
            f" csv file already exists... \n file location: {directory}cell_locations.csv \n"
        )

    logging.info("searching for tif files...")

    files, paths, folder_path = search_for_tif_files(directory)
    logging.info(f"found tif files: {files}")

    finished = []
    skipped = []
    existed = []

    for i, file in enumerate(files):
        if not ("Photocurrent" or "photocurrent") in file:
            five_hz = paths[i].split("5Hz square")
            if len(five_hz) > 1:
                try:
                    # check if five_hz is already in the first column of csv file
                    with open(
                        f"{directory}cell_locations.csv", mode="r"
                    ) as csv_file:
                        reader = csv.reader(csv_file)
                        col = [row[0] for row in reader]

                        if folder_path[i][:-11] in col:
                            logging.info(f"file {file} already in csv")
                            existed.append(file)
                            continue

                    cell_loc = select_box(paths[i])
                    logging.info(
                        f"background_location: {cell_loc}, path: {folder_path[i]}"
                    )

                    # add cell location to csv
                    with open(
                        f"{directory}cell_locations.csv", mode="a", newline=""
                    ) as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow(
                            [folder_path[i][:-11], cell_loc[0], cell_loc[1]]
                        )
                        finished.append(file)
                except Exception as exc:
                    logging.error(f"Error with file {file}", exc_info=exc)
                    skipped.append(file)
                    continue

    logging.info(
        f"\n finnished files: {finished} \n skipped files: {skipped} \n existed files: {existed}"
    )


if __name__ == "__main__":
    directory = "M:\\tnw\\ist\\do\\projects\\Neurophotonics\\Brinkslab\\People\\Xin Meng\\Code\\Python_test_TF2\\ImageAnalysis\\data"
    main(directory)
