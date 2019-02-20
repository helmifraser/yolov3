import argparse
import pandas as pd
import numpy as np
import sys

import magic

def load_csv(csv_filepath, x_filter=220, verbose=False):
    """Parses .csv into a dictionary of numpy ndarrays. Each key is the frame
        number with each entry in the ndarray corresponding to each detected
        object in the frame."""

    csv = {}
    # print("load_csv: {}".format(csv_filepath))
    with open(csv_filepath, 'r') as f:
        count = 0
        for line in f:
            count += 1
            split = line.rsplit(',')
            key = str(split[0])
            csv.setdefault(key, [])
            entries = int(len(split) - 1)

            object = []

            for idx, value in enumerate(split):
                # print(idx)
                if idx == 0:
                    continue

                if idx == entries:
                    value = value.replace("\n","")

                mod = idx % 7
                try:
                    object.append(float(value))
                except Exception as e:
                    if value == " " and verbose:
                        print("=====================")
                        print("Error loading value from .csv, ommitting")
                        print("file: {} line:{} type: {} value: {}".format(csv_filepath, count, type(value), value))
                        print("=====================")

                if mod == 0:
                    if (object[3] - object[1]) < x_filter:
                        csv.setdefault(key,[]).append(object)
                    object = []

            if len(csv[key]) != 0:
                csv[key] = np.array(csv[key])
            else:
                del csv[key]

    # for key in sorted(csv, key=int):
    #     print("{}: {}".format(key, csv[key].shape))
    return csv

def calc_centroid(points):
    check = len(points)
    if check != 4:
        print("Error in bbox_file_util.py calc_centroid(): Incorrect number of co-ordinates, " +
                "need 4 (x1, y1, x2, y2), got {} {}".format(check, points))
        return None

    x = (points[2] - points[0])/2 + points[0]
    y = (points[3] - points[1])/2 + points[1]
    centroid = (x, y)

    return centroid

def dict_centroid(csv_dict):
    """Takes the output of load_csv and returns a dictionary of centroids, in
        the same order"""

    dict = {}
    for key in sorted(csv_dict, key=int):
        for object in csv_dict[key]:
            centroid = calc_centroid(object[1:5])
            dict.setdefault(key, []).append(centroid)

        dict[key] = np.array(dict[key])

    return dict

def display(csv_file, centroid_dictionary):
    for key in sorted(csv_file, key=int):
        print("===========")
        print("Frame: {}".format(key))
        print(".csv file out: \n {}".format(csv_file[key]))
        print("Centroids of objects: \n {}".format(centroid_dictionary[key]))

def main():
    """Invoke only to test"""
    parser = argparse.ArgumentParser(description='boundary box util functions')
    parser.add_argument('-i', '--input', required=True, default=None, help='input .csv file containing bounding box labels')

    args = parser.parse_args()

    input = args.input

    csv = load_csv(input)

    dict = dict_centroid(csv)

    display(csv, dict)



if __name__ == '__main__':
    main()
