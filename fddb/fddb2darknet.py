#!/usr/bin/python3
import os
import errno
import math
from PIL import Image
import cv2
topdir = os.path.abspath(__file__+"/../")
fddb_annotations = topdir + "/FDDB-folds/FDDB-annotations.txt"
###############################
# create annotation files #
###############################
num_txt = 0
num_box = 0
with open(fddb_annotations, 'r') as annotations:
    for filepath in annotations:
        filepath = filepath.strip("\n")
        # image annotation filepath
        filepath_anno = topdir + "/images/" + filepath + ".txt"
        num_txt+=1
        with open(filepath_anno, "w+") as annotation:
            # for each ellipse in image file
            for bbox in range(int(next(annotations))):
                # find image dimensions
                img_width = 0
                img_height = 0
                with Image.open(filepath_anno.replace('.txt', '.jpg')) as img:
                    img_width, img_height = img.size
                current_line = next(annotations)
                current_line_split = current_line.split()
                major_axis_radius = float(current_line_split[0])
                minor_axis_radius = float(current_line_split[1])
                angle = float(current_line_split[2])
                center_x = float(current_line_split[3])
                center_y = float(current_line_split[4])
                #cacluate bounding box of rotated ellipse
                calc_x = math.sqrt(major_axis_radius**2 * math.cos(angle)**2 \
                                   + minor_axis_radius**2 * math.sin(angle)**2)
                calc_y = math.sqrt(major_axis_radius**2 * math.sin(angle)**2 \
                                   + minor_axis_radius**2 * math.cos(angle)**2)
                label = 0
                bbox_x = center_x / img_width
                bbox_y = center_y / img_height
                bbox_w = (2 * calc_x) / img_width
                bbox_h = (2 * calc_y) / img_height
                annotation.write("{} {} {} {} {}\n".format(label, \
                                                           bbox_x, \
                                                           bbox_y, \
                                                           bbox_w, \
                                                           bbox_h))
                num_box+=1
print("There are total {} anno txts".format(num_txt))
print("And total {} boxes".format(num_box))