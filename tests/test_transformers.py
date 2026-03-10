import unittest
import sys
import os
import torch
import numpy as np
from PIL import Image, ImageOps
from mock import mock


# append module root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onconet.transformers.image as ti
import onconet.transformers.tensor as tt

class Args():
    pass


class TestTransformers(unittest.TestCase):
    ''' Test suite for the transformers/image and transformers/tensor modules.'''
    def setUp(self):
        self.args = Args()
        self.kwargs = {}
        self.red_pixel = (255, 0, 0)
        self.green_pixel = (0, 255, 0)

    def tearDown(self):
        self.args = None
        self.kwargs = None
        self.red_pixel = None
        self.green_pixel = None

    def test_image_scale_2D(self):
        ''' Test that we get the image to a certain height and width.'''
        self.args.img_size = (200, 100)
        im = Image.new("RGB", (512, 512), "white")
        scaler = ti.Scale_2d(self.args, self.kwargs)
        expected = Image.new("RGB", (200, 100), "white")
        output = scaler(im, None)
        self.assertEqual(expected, output)

    def test_image_random_horizontal_flip(self):
        ''' Test flipping the image horizontally and keeping it the same. '''
        im = Image.new("RGB", (2, 1))
        im.putdata([self.red_pixel, self.green_pixel])
        with mock.patch('random.random', lambda: 0):
            flipper = ti.Random_Horizontal_Flip(self.args, self.kwargs)
            output = flipper(im, None)
            expected = ImageOps.mirror(im)
            self.assertEqual(expected, output)
        with mock.patch('random.random', lambda: 1):
            flipper = ti.Random_Horizontal_Flip(self.args, self.kwargs)
            output = flipper(im, None)
            self.assertEqual(im, output)

    def test_image_random_vertical_flip(self):
        ''' Test flipping the image vertically and keeping it the same.'''
        im = Image.new("RGB", (1, 2))
        im.putdata([self.red_pixel, self.green_pixel])
        flipper = ti.Random_Vertical_Flip(self.args, self.kwargs)
        with mock.patch('random.random', lambda: 0):
            output = flipper(im, None)
            expected = ImageOps.flip(im)
            self.assertEqual(expected, output)
        with mock.patch('random.random', lambda: 1):
            output = flipper(im, None)
            self.assertEqual(im, output)

    def test_image_random_crop(self):
        ''' Test cropping the image at the top.'''
        pixel = [self.red_pixel] * 16
        pixel[5], pixel[6], pixel[9], pixel[10] = [self.green_pixel] * 4
        im = Image.new("RGB", (4, 4))
        im.putdata(pixel)
        self.kwargs['w'] = 2
        self.kwargs['h'] = 2
        with mock.patch('random.randint', lambda x, y: 0):
            cropper = ti.Random_Crop(self.args, self.kwargs)
            output = cropper(im, None)
        expected = Image.new("RGB", (2, 2))
        expected.putdata([self.red_pixel, self.red_pixel, self.red_pixel, self.green_pixel])
        self.assertEqual(expected, output)

    def test_image_rotate_range(self):
        ''' Test rotating the image by 90 degrees counter-clockwise and clockwise.'''
        im = Image.new("RGB", (2, 2))
        im.putdata([self.red_pixel, self.red_pixel, self.red_pixel, self.green_pixel])
        for angle in [90, -90]:
            self.kwargs['min'] = self.kwargs['max'] = angle
            rotator = ti.Rotate_Range(self.args, self.kwargs)
            output = rotator(im, None)
            expected = im.rotate(angle)
            self.assertEqual(expected, output)

    def test_image_rotate_90(self):
        ''' Test rotating the image by 180 degrees and keeping it the same.'''
        im = Image.new("RGB", (2, 2))
        im.putdata([self.red_pixel, self.red_pixel, self.red_pixel, self.green_pixel])
        rotator = ti.Rotate_90(self.args, self.kwargs)
        with mock.patch('numpy.random.choice', lambda  x: x[2]):
            output = rotator(im, None)
            expected = Image.new("RGB", (2, 2))
            expected.putdata([self.green_pixel, self.red_pixel, self.red_pixel, self.red_pixel])
            self.assertEqual(expected, output)
        with mock.patch('numpy.random.choice', lambda  x: x[0]):
            output = rotator(im, None)
            self.assertEqual(im, output)

    def test_resize_and_pad_square_breast_on_left(self):
        '''Breast on the left half → padding should appear on the right side.'''
        self.args.img_size = (518, 518)
        # Create a tall 'I' image (200 wide x 400 high) with a bright region
        # on the LEFT half only (columns 0-99).
        img = Image.new('I', (200, 400), 0)
        bright_block = Image.new('I', (80, 300), 50000)
        img.paste(bright_block, (0, 50))  # left-side bright region

        transformer = ti.Resize_And_Pad_Square(self.args, self.kwargs)
        output = transformer(img, None)

        # Output must be exactly 518x518
        self.assertEqual(output.size, (518, 518))

        # The right column should be padded (zero), left column should be non-zero
        arr = np.array(output)
        self.assertEqual(arr[200, -1], 0,
                         "Rightmost column should be padding (0) when breast is on left")
        self.assertGreater(arr[200, 0], 0,
                           "Leftmost column should contain breast content")

    def test_resize_and_pad_square_breast_on_right(self):
        '''Breast on the right half → padding should appear on the left side.'''
        self.args.img_size = (518, 518)
        img = Image.new('I', (200, 400), 0)
        bright_block = Image.new('I', (80, 300), 50000)
        img.paste(bright_block, (120, 50))  # right-side bright region

        transformer = ti.Resize_And_Pad_Square(self.args, self.kwargs)
        output = transformer(img, None)

        self.assertEqual(output.size, (518, 518))

        arr = np.array(output)
        self.assertEqual(arr[200, 0], 0,
                         "Leftmost column should be padding (0) when breast is on right")
        self.assertGreater(arr[200, -1], 0,
                           "Rightmost column should contain breast content")

    def test_resize_and_pad_square_explicit_size_kwarg(self):
        '''Explicit size kwarg overrides args.img_size.'''
        self.args.img_size = (1664, 2048)
        local_kwargs = {'size': '256'}
        img = Image.new('I', (200, 300), 10000)
        transformer = ti.Resize_And_Pad_Square(self.args, local_kwargs)
        output = transformer(img, None)
        self.assertEqual(output.size, (256, 256))

    def test_resize_and_pad_square_pad_bottom_only(self):
        '''Wide image (wider than tall) → vertical padding goes to the bottom.'''
        self.args.img_size = (518, 518)
        img = Image.new('I', (400, 200), 30000)
        transformer = ti.Resize_And_Pad_Square(self.args, self.kwargs)
        output = transformer(img, None)
        self.assertEqual(output.size, (518, 518))
        arr = np.array(output)
        # Bottom row should be padding (0)
        self.assertEqual(arr[-1, 259], 0,
                         "Bottom row should be padding for a wide image")
        # Top row should contain image content
        self.assertGreater(arr[0, 259], 0,
                           "Top row should contain image content")

    def test_tensor_normalize_tensor_2d(self):
        ''' Test normalizing a tensor with a certain mean and a certain std deviation.'''
        self.args.img_mean = 2
        self.args.img_std = 2
        tensor = torch.IntTensor([[[4, 4, 4], [6, 6, 6]], [[8, 8, 8], [10, 10, 10]]])
        normalizer = tt.Normalize_Tensor_2d(self.args, self.kwargs)
        output = normalizer(tensor, None).numpy()
        expected = torch.IntTensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]).numpy()
        self.assertTrue(np.array_equal(expected, output))


if __name__ == '__main__':
    unittest.main()
