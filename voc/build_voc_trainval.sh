#!/bin/bash
# The folder structure is assumed to be:
#  + datasets
#     - build_voc_train.sh
#     - build_voc_test.sh
#     + train
#       + VOCdevkit
#         + VOC2012
#           + JPEGImages
#           + Annotations
#         + VOC2007
#           + JPEGImages
#           + Annotations    
CURRENT_DIR=$(pwd)

echo "Download VOC2012_trainval"
# wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
echo "Download VOC2007_trainval"
# wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar

echo "create trainval folder"
mkdir "$(pwd)/trainval"
echo "untar"
# tar xvf VOCtrainval_06-Nov-2007.tar -C "$(pwd)/trainval"
# tar xvf VOCtrainval_11-May-2012.tar -C "$(pwd)/trainval"
echo "build voc trainval"
# python3 create_voc_darknet.py $CURRENT_DIR trainval merged "$CURRENT_DIR/voc_label_map.json"

echo "zip dataset file: for colab ..."
# zip -rj train2012.zip trainval/VOCdevkit/VOC2012/Darknet_Annos trainval/VOCdevkit/VOC2012/JPEGImages
# zip -rj train2007.zip trainval/VOCdevkit/VOC2007/Darknet_Annos trainval/VOCdevkit/VOC2007/JPEGImages

echo "eda dataset"
python3 create_voc_stat.py $CURRENT_DIR trainval merged "$CURRENT_DIR/voc_label_map.json"

echo "done"