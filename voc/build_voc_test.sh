#!/bin/bash
# The folder structure is assumed to be:
#  + datasets
#     - build_voc_train.sh
#     - build_voc_test.sh
#     + test
#       + VOCdevkit
#         + VOC2007
#           + JPEGImages
#           + Annotations
CURRENT_DIR=$(pwd)

echo "Download vod 2007 test"
# wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar # VOC2007 test set

echo "mkdir"
# mkdir "$(pwd)/test"

echo "untar"
# tar xvf VOCtest_06-Nov-2007.tar -C "$(pwd)/test"

echo "create voc labelmap"
python3 create_voc_labelmap.py

echo "build voc test"
python3 create_voc_darknet.py $CURRENT_DIR 'test' 'VOC2007' "$CURRENT_DIR/voc_label_map.json"

echo "zip dataset file: for colab ..."
# zip -rj test2007.zip test/VOCdevkit/VOC2007/Darknet_Annos test/VOCdevkit/VOC2007/JPEGImages


echo "eda dataset"
python3 create_voc_stat.py $CURRENT_DIR 'test' 'VOC2007' "$CURRENT_DIR/voc_label_map.json"

echo "done"