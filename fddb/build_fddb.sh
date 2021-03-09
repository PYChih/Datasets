#!/bin/bash
# download fddb original dataset
# download fddb dataset from original websites
echo "download original, unannotated set of images"
wget http://vis-www.cs.umass.edu/fddb/originalPics.tar.gz
echo "download face annotations"
wget http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz
echo "download README"
wget http://vis-www.cs.umass.edu/fddb/README.txt

IMAGE_FILE="originalPics.tar.gx"
LABEL_FILE="FDDB-folds.tgz"
echo "uncompress" "$IMAGE_FILE" "$LABEL_FILE"

tar -xzvf "$LABEL_FILE"
mkdir "images" && tar -xzvf "originalPics.tar.gz" -C "images"
echo "create FDDB-paths.txt and FDDB-annotations.txt"
find FDDB-folds -type f -regex ".*[0-9]-.*txt" -exec cat {} >> FDDB-folds/FDDB-annotations.txt \;
find FDDB-folds -type f -regex ".*[0-9]+.txt" -exec cat {} >> FDDB-folds/FDDB-paths.txt \;
echo "done"

echo 'creating annotation file: [imagename].replace(jpg, txt) ...'
python3 ./fddb2darknet.py
echo "done"

echo "zip dataset file: fddb_clean.zip for colab ..."
cp FDDB-folds/FDDB-paths.txt images/FDDB-paths.txt
zip -r fddb_clean images/
echo "done"
