
"""Convert VOC_XML to darknet_txt for object_detection.
Example usage:
    python3 create_voc_darknet.py \
            $CURRENT_DIR \
            train \
            merged \
            "$CURRENT_DIR/voc_label_map.json"
"""
import logging
import os
from lxml import etree
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("top_dir", type = str,
                    help="Root directory to VOC dataset")
parser.add_argument("set", type = str, default = "train",
                    help="train or test")
parser.add_argument("year", type = str, default = "VOC2012",
                    help="VOC2012 / VOC2007 or merged")
parser.add_argument("label_map_path", type = str,
                    help="Path to label map *.json")
SETS = ['train', 'val', 'trainval', 'test']
YEARS = ['VOC2007', 'VOC2012', 'merged']

def read_examples_list(path):
    """Read list of training or validation examples.
    The file is assumed to contain a single example per line where the first
    token in the line is an identifier that allows us to find the image and
    annotation xml for that example.
    For example, the line:
    xyz 3
    would allow us to find files xyz.jpg and xyz.xml (the 3 would be ignored).

    Args:
      path: absolute path to examples list file.

    Returns:
      list of example identifiers (strings).
    """
    with open(path) as fid:
        lines = fid.readlines()
    return [line.strip().split(' ')[0] for line in lines]

def recursive_parse_xml_to_dict(xml):
    """Recursively parses XML contents to python dict.
    We assume that `object` tags are the only ones that can appear
    multiple times at the same level of a tree.

    Args:
      xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
      Python dictionary holding XML contents.
    """
    if len(xml) == 0:
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = recursive_parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}

def dict_to_darknet_example(data,
                            label_map_dict,
                            ignore_difficult_instances=False):
    """Convert XML derived dict to darknet proto.
    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
        data: dict holding PASCAL XML fields for a single image (obtained by
          running recursive_parse_xml_to_dict)
        label_map_dict: A map from string label names to integers ids.
        ignore_difficult_instances: Whether to skip difficult instances in the
          dataset  (default: False).
        image_subdirectory: String specifying subdirectory within the
          PASCAL dataset directory holding the actual image data.

    Returns:
        example: string

    """
    width = int(data['size']['width'])
    height = int(data['size']['height'])
    anno = str()
    if 'object' in data:
        for obj in data['object']:
            difficult = bool(int(obj['difficult']))
            if ignore_difficult_instances and difficult:
                continue
            label = label_map_dict[obj['name']]
            xmin = float(obj['bndbox']['xmin'])
            xmax = float(obj['bndbox']['xmax'])
            ymin = float(obj['bndbox']['ymin'])
            ymax = float(obj['bndbox']['ymax'])
            bbox_x = ((xmin + xmax) / 2) / width
            bbox_y = ((ymin + ymax) / 2) / height
            bbox_w = ((xmax - xmin)) / width
            bbox_h = ((ymax - ymin)) / height
            anno += "{} {} {} {} {}\n".format(label,
                                              bbox_x,
                                              bbox_y,
                                              bbox_w,
                                              bbox_h)
    return anno

def main(args):
    if args.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))
    if args.year not in YEARS:
        raise ValueError('year must be in : {}'.format(YEARS))
    top_dir = args.top_dir
    data_dir = os.path.join(top_dir, args.set, 'VOCdevkit')
    years = ['VOC2007', 'VOC2012']
    if args.year != 'merged':
        years = [args.year]
    print("reading label_map from {}".format(args.label_map_path))
    with open(args.label_map_path, 'r') as f:
        label_map = json.load(f)

    for year in years:
        logging.info('Reading from {} {} dataset.'.format(year, args.set))
        examples_path = os.path.join(data_dir, year, 'ImageSets', 'Main',
                                     'aeroplane_' + args.set + '.txt')
        annotations_dir = os.path.join(data_dir, year, 'Annotations')
        write_dir = os.path.join(data_dir, year, "Darknet_Annos")
        if not os.path.isdir(write_dir):
            os.mkdir(write_dir)
        examples_list = read_examples_list(examples_path)
        for idx, example in enumerate(examples_list):
            if idx % 100 == 0:
                logging.info('On image {} of {}', idx, len(examples_list))
            path = os.path.join(annotations_dir, example + '.xml')
            write_path = os.path.join(write_dir, example + '.txt')
            with open(path, 'r') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = recursive_parse_xml_to_dict(xml)['annotation']
            darknet_example = dict_to_darknet_example(data, label_map)
            with open(write_path, 'w') as fw:
                fw.write(darknet_example)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)



