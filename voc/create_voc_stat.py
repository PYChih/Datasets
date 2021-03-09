"""understanding basic information for this dataset
1. total number of anno image
2. total number of box
3. number of box for each class
3. visualize and write image
"""
import os
from PIL import Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import numpy as np
import json
import random
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

def read_single_img(img_dir, name):
    img_path = os.path.join(img_dir, name + '.jpg')
    img = Image.open(img_path, 'r')
    return img

def read_single_anno(anno_dir, name):
    anno_path = os.path.join(anno_dir, name + '.txt')
    with open(anno_path) as f:
        lines = f.readlines()
    return lines

def get_classidx2labelname_dict(label_map_path):
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    classidx2labelname = {v:k for k, v in label_map.items()}
    return classidx2labelname

def write_data_name(anno_dir, set_type, year):
    txt_file_name = "{}_{}.txt".format(year, set_type)
    anno_file_list = os.listdir(anno_dir)
    with open(txt_file_name, 'w') as f:
        file_cnt = 0
        for filename in anno_file_list:
            filename = filename.strip('.txt')
            file_cnt += 1
            f.write(filename + '\n')
    return txt_file_name, file_cnt

def stat_box(data_txt, data_dict, anno_dir, classidx2labelname):
    with open(data_txt, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        ann_list = read_single_anno(anno_dir, line)
        for ann in ann_list:
            ann = ann.strip('\n')
            box_list = ann.split()
            class_idx = int(box_list[0])
            labelname = classidx2labelname[class_idx]
            if labelname in data_dict:
                data_dict[labelname]+=1
            else:
                data_dict[labelname]=1
    return data_dict

def draw_bounding_box_on_image(image,
                               xmin,
                               ymin,
                               xmax,
                               ymax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
    """Adds a bounding box to an image.
    
    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.
    
    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.
    
    Args:
    image: a PIL.Image object.
    xmin: xmin of bounding box.
    ymin: ymin of bounding box.
    xmax: xmax of bounding box.
    ymax: ymax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                     (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
     ymin, xmin, ymax, xmax as relative to the image. Otherwise treat
     coordinates as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    if thickness > 0:
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
                  width = thickness,
                  fill = color)
    font = ImageFont.load_default()
    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
    
    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                              text_bottom)],
            fill=color)
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)
        text_bottom -= text_height - 2 * margin

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
    classidx2labelname = get_classidx2labelname_dict(args.label_map_path)
    box_stat_dict = {}
    total_anno_cnt = 0
    for year in years:
        anno_dir = os.path.join(data_dir, year, 'Darknet_Annos')
        filename, cnt = write_data_name(anno_dir, args.set, year)
        print("total {} annotations in {} dataset".format(cnt, filename.strip('.txt')))
        total_anno_cnt+=cnt
        box_stat_dict = stat_box(filename, box_stat_dict, anno_dir, classidx2labelname)
    box_stat_dict['total_num_images'] = total_anno_cnt
    with open('{}_{}_stat_info.json'.format(args.year, args.set), 'w') as f:
        json.dump(box_stat_dict, f, indent=4)
    # visualize
    imwrite_dir = os.path.join(args.top_dir, "example_img")
    if not os.path.isdir(imwrite_dir):
        os.mkdir(imwrite_dir)
    anno_name_list = os.listdir(anno_dir)
    img_dir = anno_dir.replace('Darknet_Annos', 'JPEGImages')
    for i in range(5):
        k = random.randint(0, len(anno_name_list))
        name = anno_name_list[k].strip('.txt')
        ann_list = read_single_anno(anno_dir, name)
        img = read_single_img(img_dir, name)
        for ann in ann_list:
            ann = ann.strip('\n')
            box_list = ann.split()
            class_idx = int(box_list[0])
            labelname = classidx2labelname[class_idx]
            bbox_x = float(box_list[1])
            bbox_y = float(box_list[2])
            bbox_w = float(box_list[3])
            bbox_h = float(box_list[4])
            xmin = bbox_x - (bbox_w / 2)
            ymin = bbox_y - (bbox_h / 2)
            xmax = bbox_x + (bbox_w / 2)
            ymax = bbox_y + (bbox_h / 2)
            draw_bounding_box_on_image(img,
                                       xmin,
                                       ymin,
                                       xmax,
                                       ymax,
                                       color='red',
                                       thickness=4,
                                       display_str_list=[labelname],
                                       use_normalized_coordinates=True)
        img.save(os.path.join(imwrite_dir, name+'.jpg'))
        
        
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)