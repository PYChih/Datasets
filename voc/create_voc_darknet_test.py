"""Test for create_voc_darknet.py."""
import unittest
import os
import create_voc_darknet
import tempfile

class TestCreate_voc_darknet(unittest.TestCase):
    def test_read_examples_list(self):
        example_list_data = "example1 1\nexample2 2"
        example_list_path = os.path.join(tempfile.gettempdir(), 'examples.txt')
        with open(example_list_path, 'w') as f:
            f.write(example_list_data)
        
        examples = create_voc_darknet.read_examples_list(example_list_path)
        self.assertListEqual(['example1', 'example2'], examples)
    
    def test_dict_to_darknet_example(self):
        image_file_name = 'tmp_image.jpg'
        data = {
            'folder': '',
            'filename': image_file_name,
            'size': {
                'height': 320,
                'width': 240,
            },
            'object':[
                {
                    'difficult':1,
                    'bndbox':{
                        'xmin':64,
                        'ymin':32,
                        'xmax':128,
                        'ymax':256,
                    },
                    'name':'person',
                    'truncated': 0,
                    'pose':'',
                },
            ],
        }

        label_map_dict = {
            'background': 0,
            'person':1,
            'notperson':2,
        }
        example = create_voc_darknet.dict_to_darknet_example(
            data, label_map_dict
        )
        self.assertEqual(example.split(), ['1', str((128+64)/2/240), str((256+32)/2/320), str(((128-64))/240), str(((256-32))/320)])

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCreate_voc_darknet)
    unittest.TextTestRunner(verbosity=2).run(suite)