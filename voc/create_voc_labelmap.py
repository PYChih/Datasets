import json

PASCAL_CLASSES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
]

def get_voc_label_map_dict(classes_name, include_background=False):
    """create a map from string label names to integers ids.
    Args:
        classes_name: list of classname
        include_background: bool
    Returns:
        
    """
    label_map = {}
    for idx, class_name in enumerate(classes_name):
        if include_background:
            label_map[class_name] = idx+1
        else:
            label_map[class_name] = idx
    return label_map

if __name__ == '__main__':
    voc_label_map = get_voc_label_map_dict(PASCAL_CLASSES)
    with open('voc_label_map.json', 'w') as f:
        json.dump(voc_label_map, f, indent=4)