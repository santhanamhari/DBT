import torchvision
import random
import numpy as np
from PIL import Image, ImageStat, ImageOps, ImageFile
import pdb
from onconet.transformers.factory import RegisterImageTransformer
from onconet.transformers.abstract import Abstract_transformer
from onconet.utils.region_annotation import flip_region_coords_left_right, flip_region_coords_top_bottom, rotate_region_coords_angle, make_region_annotation_blank
import warnings

ImageFile.LOAD_TRUNCATED_IMAGES = True
CLASS_NOT_SUPPORT_REGION_WARNING = "{} does not support region annotations! Bounding box coordinates removed to prevent incorrect behavior"

class CordRescaler():
    def __init__(self, scaled_w, scaled_h):
        self.scaled_w = scaled_w
        self.scaled_h = scaled_h

    def get_xy(self, original_w, original_h, x, y):
        '''
        Compute the new x,y coordinates in an image that was rescaled
        from original_w  X original_h
        to self.scaled_w X self.scaled_h
        '''
        related_x = float(x) / original_w
        related_y = float(y) / original_h
        new_x = int(related_x * self.scaled_w)
        new_y = int(related_y * self.scaled_h)
        return new_x, new_y


class Point:
    def __init__(self, xcoord, ycoord):
        self.x = xcoord
        self.y = ycoord


class Rectangle:
    def __init__(self, bottom_left, top_right):
        self.bottom_left = bottom_left
        self.top_right = top_right

    def intersects(self, other):
        return not (self.top_right.x < other.bottom_left.x
                    or self.bottom_left.x > other.top_right.x
                    or self.top_right.y < other.bottom_left.y
                    or self.bottom_left.y > other.top_right.y)


def in_overlays(x1, y1, patch_width, patch_height, overlays, scaler,
                img_origin_size):
    for overlay in overlays:
        if in_overlay(x1, y1, patch_width, patch_height, overlay, scaler,
                      img_origin_size):
            return True

    return False


def in_overlay(x1, y1, patch_width, patch_height, overlay, scaler,
               img_origin_size):
    patch = Rectangle(
        Point(x1, y1 + patch_height), Point(x1 + patch_width, y1))
    x, y = scaler.get_xy(img_origin_size[0], img_origin_size[1], overlay['boundary']['min_x'],
                      overlay['boundary']['max_y'])
    bottom_left = Point(x, y)
    x, y = scaler.get_xy(img_origin_size[0], img_origin_size[1], overlay['boundary']['max_x'],
                      overlay['boundary']['min_y'])
    top_right = Point(x, y)
    overlay = Rectangle(bottom_left, top_right)
    return patch.intersects(overlay)


@RegisterImageTransformer("extract_patch")
class ExtractPatch(Abstract_transformer):
    '''
    Extract a patch based on the label and overlays that are in the
    additional data dict.
    The size of the patch will be w/h - based on kwargs.
    The patch location will be decided by the label:
    0 - randomly out of the all the overlay polygons
    else - center of overlay
    '''

    def __init__(self, args, kwargs):
        super(ExtractPatch, self).__init__()
        assert len(kwargs.keys()) == 1
        self.patch_width, self.patch_height = args.patch_size
        self.zoom = float(kwargs['z'])
        self.width, self.height = args.img_size
        self.cord_rescaler = CordRescaler(self.width, self.height)
        self.patch_scaler = torchvision.transforms.Resize( ( self.patch_height, self.patch_width))
        self.random_cropper = torchvision.transforms.RandomCrop( (self.patch_width, self.patch_height) )

    def __call__(self, img, additional=None):
        label = additional['label']
        zoom_factor = np.random.uniform(
                                low = 1 - self.zoom, high = 1 + self.zoom )
        patch_width =  int(self.patch_width * zoom_factor)
        patch_height = int(self.patch_height * zoom_factor)
        if label == 0:
            x1, y1 = (0, 0)
            patch_in_overlay = True
            while patch_in_overlay:
                x1 = random.randint(0, self.width - patch_width)
                y1 = random.randint(0, self.height - patch_height)
                patch_in_overlay = in_overlays(
                    x1, y1, patch_width, patch_height,
                    additional['all_overlays'], self.cord_rescaler,
                    (additional['width'], additional['height']))

            return self.patch_scaler( img.crop((x1 - patch_width//2 , y1 - patch_width//2, x1 + patch_width//2,
                             y1 + patch_height//2)) )
        else:

            x1, y1 = self.cord_rescaler.get_xy(additional['width'],
                                               additional['height'],
                                               additional['boundary']['center_x'],
                                               additional['boundary']['center_y'])

            if zoom_factor > 1:
                return self.random_cropper( img.crop((x1 - patch_width//2 , y1 - patch_width//2, x1 + patch_width//2,
                             y1 + patch_height//2)) )
            else:
                return self.patch_scaler( img.crop((x1 - patch_width//2 , y1 - patch_width//2, x1 + patch_width//2,
                             y1 + patch_height//2)) )


@RegisterImageTransformer("scale_2d")
class Scale_2d(Abstract_transformer):
    '''
        Given PIL image, enforce its some set size
        (can use for down sampling / keep full res)
    '''

    def __init__(self, args, kwargs):
        super(Scale_2d, self).__init__()
        assert len(kwargs.keys()) == 0
        width, height = args.img_size
        self.set_cachable(width, height)
        self.transform = torchvision.transforms.Resize((height, width))

    def __call__(self, img, additional=None):
        return self.transform(img.convert('I'))

@RegisterImageTransformer("scale_2d_with_fixed_aspect_ratio")
class Scale_2d_With_Fixed_Aspect_Ratio(Abstract_transformer):
    '''
        Given PIL image, enforce its some set size
        (can use for down sampling / keep full res)

        Does it in 3 steps:
        1) determine if left or right (similar to align left)
    '''

    def __init__(self, args, kwargs):
        super(Scale_2d_With_Fixed_Aspect_Ratio, self).__init__()
        assert len(kwargs.keys()) == 0
        width, height = args.img_size
        self.set_cachable(width, height)

        self.aspect_ratio = float(height) / float(width)

        self.scale_transform = torchvision.transforms.Resize((height, width))

        # Create black image
        mask_r = Image.new('1', args.img_size)
        # Paint right side in white
        mask_r.paste(1, ((mask_r.size[0] *3 // 4), 0, mask_r.size[0],
                         mask_r.size[1]))
        mask_l = mask_r.transpose(Image.FLIP_LEFT_RIGHT)

        self.mask_r = mask_r
        self.mask_l = mask_l
        self.black = Image.new('I', args.img_size)

    def __call__(self, img, additional=None):
        width, height = img.size
        expected_width = int( height / self.aspect_ratio)
        if expected_width != width:
            assert width < expected_width
            pad_delta = expected_width - width
            left_pad = (pad_delta, 0, 0, 0)
            right_pad = (0, 0, pad_delta, 0)
            # Figure out if pad left or right.
            left = img.copy()
            left.paste(self.black, mask = self.mask_l)
            left_sum = np.array(left.getdata()).sum()
            right = img.copy()
            right.paste(self.black, mask = self.mask_r)
            right_sum = np.array(right.getdata()).sum()
            if right_sum > left_sum:
                pad = left_pad
            else:
                pad = right_pad
            img = ImageOps.expand(img, pad)
            new_aspect_ratio = float(img.size[1]) / img.size[0]
            assert new_aspect_ratio == self.aspect_ratio
        return self.scale_transform(img)


@RegisterImageTransformer("rand_hor_flip")
class Random_Horizontal_Flip(Abstract_transformer):
    '''
    torchvision.transforms.RandomHorizontalFlip wrapper
    '''

    def __init__(self, args, kwargs):
        super(Random_Horizontal_Flip, self).__init__()
        self.args = args
        assert len(kwargs.keys()) == 0

    def __call__(self, img, additional=None):
        if random.random() < 0.5:
            flip_region_coords_left_right(additional)
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


@RegisterImageTransformer("rand_ver_flip")
class Random_Vertical_Flip(Abstract_transformer):
    '''
    random vertical flip.
    '''

    def __init__(self, args, kwargs):
        super(Random_Vertical_Flip, self).__init__()
        self.args = args
        assert len(kwargs.keys()) == 0

    def __call__(self, img, additional=None):
        if random.random() < 0.5:
            flip_region_coords_top_bottom(additional)
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


@RegisterImageTransformer("random_crop")
class Random_Crop(Abstract_transformer):
    '''
        torchvision.transforms.RandomCrop wrapper
        size of cropping will be decided by the 'h' and 'w' kwargs.
        'padding' kwarg is also available.
    '''

    def __init__(self, args, kwargs):
        super(Random_Crop, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert kwargs_len in [2,3]
        size = (int(kwargs['h']), int(kwargs['w']))

        padding = int(kwargs['padding']) if 'padding' in kwargs else 0
        self.transform = torchvision.transforms.RandomCrop(size, padding)


    def __call__(self, img, additional=None):
        if self.args.use_region_annotation and additional is not None and 'region_annotation' in additional:
            warnings.warn(CLASS_NOT_SUPPORT_REGION_WARNING.format(self.__class__))
            make_region_annotation_blank(additional)

        return self.transform(img)

@RegisterImageTransformer("rotate_range")
class Rotate_Range(Abstract_transformer):
    '''
    Rotate image counter clockwise by random
    kwargs['min'] - kwargs['max'] degrees.

    Example: 'rotate/min=-20/max=20' will rotate by up to +/-20 deg
    '''

    def __init__(self, args, kwargs):
        super(Rotate_Range, self).__init__()
        assert len(kwargs.keys()) == 2
        self.max_angle = int(kwargs['max'])
        self.min_angle = int(kwargs['min'])

    def __call__(self, img, additional=None):
        angle = random.randint(self.min_angle, self.max_angle)
        rotate_region_coords_angle(angle, additional)
        return img.rotate(angle)


@RegisterImageTransformer("rotate_90")
class Rotate_90(Abstract_transformer):
    '''
    Rotate image by 0/90/180/270 degrees randomly.
    '''

    def __init__(self, args, kwargs):
        super(Rotate_90, self).__init__()
        assert len(kwargs.keys()) == 0
        self.args = args
        self.rotations = [
            0, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270
        ]

        self.rotation_to_angle = {
            0:0, Image.ROTATE_90:90, Image.ROTATE_180:180, Image.ROTATE_270:270
        }

    def __call__(self, img, additional=None):
        rotation = np.random.choice(self.rotations)
        if rotation:
            angle = self.rotation_to_angle[rotation]
            rotate_region_coords_angle(angle, additional)
            return img.transpose(rotation)
        else:
            return img


@RegisterImageTransformer("align_to_left")
class Align_To_Left(Abstract_transformer):
    '''
    Aligns all images s.t. the breast will face left.
    Note: this should be applied after the scaling since the mask
    is the size of args.img_size.
    torchvision.transforms.RandomHorizontalFlip wrapper
    '''

    def __init__(self, args, kwargs):
        super(Align_To_Left, self).__init__()
        assert len(kwargs.keys()) == 0

        self.set_cachable(args.img_size)

        # Create black image
        mask_r = Image.new('1', args.img_size)
        # Paint right side in white
        mask_r.paste(1, ((mask_r.size[0] *3 // 4), 0, mask_r.size[0],
                         mask_r.size[1]))
        mask_l = mask_r.transpose(Image.FLIP_LEFT_RIGHT)

        self.mask_r = mask_r
        self.mask_l = mask_l
        self.black = Image.new('I', args.img_size)

    def __call__(self, img, additional=None):
        left = img.copy()
        left.paste(self.black, mask = self.mask_l)
        left_sum = np.array(left.getdata()).sum()
        right = img.copy()
        right.paste(self.black, mask = self.mask_r)
        right_sum = np.array(right.getdata()).sum()
        if right_sum > left_sum:
            flip_region_coords_left_right(additional)
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return img

@RegisterImageTransformer("resize_and_pad_square")
class Resize_And_Pad_Square(Abstract_transformer):
    '''
    Resize a mammography / DBT frame with preserved aspect ratio so it fits
    within target_size x target_size, then pad to an exact square.

    Padding rules
    -------------
    - Vertical   : pad on the **bottom** only (top stays at the image edge).
    - Horizontal : pad on the side **opposite** the breast so the breast
                   region stays flush against its original edge.

    The breast side is determined by counting pixels above *threshold* in the
    left vs right half of the resized image.

    Accepted kwargs (all optional)
    --------------------------------
    size            int   – target square side length; defaults to
                            max(args.img_size) when img_size is a 2-tuple,
                            or args.img_size directly when it is a scalar.
    threshold       float – pixel value below which a pixel is considered
                            background (default 0).
    pad_value       int   – fill value used for padding (default 0).
    pad_bottom_only bool  – when True (default) vertical pad goes to bottom;
                            when False it is split top/bottom (centred).
    debug           bool  – when True, save debug images to /tmp (default False).
    '''

    def __init__(self, args, kwargs):
        super(Resize_And_Pad_Square, self).__init__()

        # ------------------------------------------------------------------
        # Determine target size
        # ------------------------------------------------------------------
        if 'size' in kwargs:
            self.size = int(kwargs['size'])
        else:
            img_size = args.img_size
            if isinstance(img_size, (list, tuple)):
                self.size = max(img_size)
            else:
                self.size = int(img_size)

        self.threshold = float(kwargs['threshold']) if 'threshold' in kwargs else 0.0
        self.pad_value = int(kwargs['pad_value']) if 'pad_value' in kwargs else 0

        # Accept both bool and "true"/"false" strings (kwargs from CLI are strings)
        raw_pb = kwargs.get('pad_bottom_only', True)
        if isinstance(raw_pb, str):
            self.pad_bottom_only = raw_pb.lower() not in ('false', '0', 'no')
        else:
            self.pad_bottom_only = bool(raw_pb)

        raw_debug = kwargs.get('debug', False)
        if isinstance(raw_debug, str):
            self.debug = raw_debug.lower() in ('true', '1', 'yes')
        else:
            self.debug = bool(raw_debug)

        self.set_cachable(self.size)

    def __call__(self, img, additional=None):
        # ------------------------------------------------------------------
        # Normalise to PIL mode 'I' (32-bit signed integer grayscale)
        # ------------------------------------------------------------------
        if img.mode == 'I;16':
            arr = np.frombuffer(img.tobytes(), dtype=np.uint16).reshape(
                img.size[1], img.size[0])
            img = Image.fromarray(arr.astype(np.int32), mode='I')
        else:
            img = img.convert('I')

        orig_w, orig_h = img.size
        size = self.size

        # ------------------------------------------------------------------
        # Resize with preserved aspect ratio
        # ------------------------------------------------------------------
        scale = min(size / orig_w, size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        resized = img.resize((new_w, new_h), Image.LANCZOS)

        # ------------------------------------------------------------------
        # Determine breast side on the resized image
        # ------------------------------------------------------------------
        arr = np.array(resized)
        mid_x = new_w // 2
        left_count = int((arr[:, :mid_x] > self.threshold).sum())
        right_count = int((arr[:, mid_x:] > self.threshold).sum())
        breast_on_left = left_count >= right_count

        # ------------------------------------------------------------------
        # Compute padding amounts
        # ------------------------------------------------------------------
        pad_w = size - new_w  # total horizontal padding needed
        pad_h = size - new_h  # total vertical padding needed

        # Horizontal: opposite side to breast
        if breast_on_left:
            pad_left, pad_right = 0, pad_w
        else:
            pad_left, pad_right = pad_w, 0

        # Vertical
        if self.pad_bottom_only:
            pad_top, pad_bottom = 0, pad_h
        else:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top

        if self.debug:
            resized.save('/tmp/resize_pad_debug.png')

        # ImageOps.expand border order: (left, top, right, bottom)
        padded = ImageOps.expand(
            resized,
            (pad_left, pad_top, pad_right, pad_bottom),
            fill=self.pad_value,
        )
        return padded


@RegisterImageTransformer("grayscale")
class Grayscale(Abstract_transformer):
    '''
    Given PIL image, converts it to grayscale
    with args.num_chan channels.
    '''

    def __init__(self, args, kwargs):
        super(Grayscale, self).__init__()
        assert len(kwargs.keys()) == 0
        self.set_cachable(args.num_chan)

        self.transform = torchvision.transforms.Grayscale(args.num_chan)

    def __call__(self, img, additional=None):
        return self.transform(img)
