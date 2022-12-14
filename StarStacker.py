from enum import Enum
from pathlib import Path

from detectors.template_detector import *
from detectors.blob_detector import *

from processor.macro_processor import *
from processor.ncc_processor import *
from processor.median_processor import *
from processor.normalize_processor import *
from processor.shift_processor import *

from matcher.icp_matcher import *
from matcher.brute_force_matcher import *

from helper.opencv_helper import *


class StarDetectionMethod(Enum):
    Blob = 1
    Ncc = 2


class StarStackerParameter:
    def __init__(self):
        self.__star_detector = StarDetectionMethod.Ncc
        self.__path = "./"

    @property
    def star_detector(self):
        return self.__star_detector

    @star_detector.setter
    def star_detector(self, value):
        self.__star_detector = value

    @property
    def path(self):
        return self.__path

    @path.setter
    def path(self, value):
        self.__path = value

    @property
    def ref_image(self):
        return self.__ref_image

    @ref_image.setter
    def ref_image(self, value):
        self.__ref_image = value


class StarStackerException(Exception):
    def __init__(self):
        pass

class StarStacker:
    def __init__(self):
        self.__star_detector = StarDetectionMethod.Ncc
        self.__param = None
        self.__path = None
        self.__preprocessor = None
        self.__detector = None


    def __check_parameter(self, param : StarStackerParameter):
        self.__param = param

        # validate parameters
        path = Path(param.path)
        if not path.exists:
            raise NotADirectoryError(f'Directory {path} does not exist.') 
        self.__path = path

        file = self.__path / param.ref_image
        if not file.is_file():
            raise FileNotFoundError(f'Reference image file {param.ref_file} does not exists.')
        self.__ref_file = file


    def __detect_stars(self, image : np.array):
        ''' Detect stars and return coordinate list.'''    
        stars = self.__detector.search(image)
        coords = np.array([(s[0], s[1]) for s in stars]).T
        return coords


    def __show_anot_images(self, ref_img_name, orig_image, ref_points, img_name, image, points, transform, windows_size = 1500):
        left_image = orig_image.copy()
        img_height, img_width = left_image.shape[:2]

        box_size = img_height / 50
        for pos in ref_points.T:
            x, y = pos[:2]
            cv2.rectangle(left_image, (int(x - box_size/2), int(y - box_size/2)), (int(x + box_size/2), int(y + box_size/2)), (0,255,0), 6)

        right_image = image.copy()
        box_size = img_height / 100
        for pos in points.T:
            x, y = pos[:2]
            cv2.rectangle(right_image, (int(x - box_size/2), int(y - box_size/2)), (int(x + box_size/2), int(y + box_size/2)), (0,0,255), -1)

        top_image = np.concatenate((left_image, right_image), axis=1)
        img_height, img_width = top_image.shape[:2]
        scale = windows_size / img_width

        top_image = cv2.resize(top_image, (int(scale*img_width), int(scale*img_height)))

        #
        # Bottom Row
        #

        img_height, img_width = orig_image.shape[:2]    
        left_image = np.zeros((img_height, img_width, 3), np.uint8)  
        right_image = np.zeros((img_height, img_width, 3), np.uint8)  
        box_size = img_height / 50
        for pos in ref_points.T:
            x, y = pos[:2]
            cv2.rectangle(left_image, (int(x - box_size/2), int(y - box_size/2)), (int(x + box_size/2), int(y + box_size/2)), (0,255,0), 6)

        box_size = img_height / 100
        for pos in points.T:
            x, y = pos[:2]
            cv2.rectangle(left_image, (int(x - box_size/2), int(y - box_size/2)), (int(x + box_size/2), int(y + box_size/2)), (0, 0,255), -1)

        if not transform is None:
            points_reg = np.array([points.T], copy=True).astype(np.float32)
            points_reg = cv2.transform(points_reg, transform)

            box_size = img_height / 50
            for pos in ref_points.T:
                x, y = pos[:2]
                cv2.rectangle(right_image, (int(x - box_size/2), int(y - box_size/2)), (int(x + box_size/2), int(y + box_size/2)), (0,255,0), 6)

            box_size = img_height / 100
            for pos in points_reg[0]:
                x, y = pos[:2]
                cv2.rectangle(right_image, (int(x - box_size/2), int(y - box_size/2)), (int(x + box_size/2), int(y + box_size/2)), (0, 0,255), -1)

        bottom_image = np.concatenate((left_image, right_image), axis=1)
        img_height, img_width = bottom_image.shape[:2]
        scale = windows_size / img_width

        bottom_image = cv2.resize(bottom_image, (int(scale*img_width), int(scale*img_height)))

        overview = np.concatenate((top_image, bottom_image), axis=0)

        win_height, win_width = overview.shape[:2]
        text_x = 10
        text_y = 30
        color = (255,255,255)
        thickness = 1
        cv2.putText(overview, f'Reference: {ref_img_name}', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness, cv2.LINE_AA)
        cv2.putText(overview, f'Current: {img_name}', ((win_width>>1) +  text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness, cv2.LINE_AA)
        cv2.putText(overview, f'Point Clouds', (text_x, (win_height>>1)+text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness, cv2.LINE_AA)
        cv2.putText(overview, f'Matched Results', ((win_width>>1) +  text_x, (win_height>>1)+text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness, cv2.LINE_AA)

        cv2.imshow('Detection Result', overview)
        cv2.waitKey(50)


    def __align_stack(self):
        img_ref_proc, img_ref_orig = imread(str(self.__ref_file), self.__preprocessor)
      
        ref_points = self.__detect_stars(img_ref_proc)
        print(f'{self.__detector.name} detected {ref_points.shape[1]} stars in the reference image.')

        ct = 0
        ct_fail = 0
        transform = None

        # offset of the last processed image to the ref image. Is used as a starting point 
        # for point cloud matching to increase chances of convergence.
        offset_ref_image = np.array([0.0, 0.0])
        
        matcher_bf = BruteForceMatcher()
        matcher_icp = IcpMatcher(100, 10)

        pathlist = self.__path.glob('**/*.*')

        for path in pathlist:
            if path.name == self.__ref_file.name:
                continue

            if not path.suffix.lower() in ['.cr2', 'jpg']:
                continue

            # read the image
            img_orig, _ = imread(str(path))

            ct += 1
            retry = 0

            while True:
                try:
                    # Read images, preprocess and shift image by last offset. This makes the 
                    # point cloud matching more likely to succeed. Because images in a series 
                    # are close to one another and if the last image could be matched the 
                    # offset is almost correct.
                    shift_processor = ShiftProcessor(offset_ref_image)
                    img_proc = improcess(img_orig, [self.__preprocessor, shift_processor])
                    img_orig_shift = shift_image(img_orig, offset_ref_image)                    
                    points = self.__detect_stars(img_proc)            

                    # remove all detections too close to the edges of the shifted image
                    # Based on the template the template matcher has a preference to detecting 
                    # points right at the border
                    h, w = img_proc.shape[:2]
                    ul = np.array([10.0, 10.0]) + offset_ref_image
                    lr = np.array([w-10, h-10]) + offset_ref_image
                    good_matches = np.all(np.logical_and(points.T > ul, points.T < lr), axis=1)
                    points = points.T[good_matches].T

                    transform = matcher_icp.match(ref_points, points)
                    offset_ref_image += transform[:2, 2]

                    #
                    # matching succeeded, save registered image
                    #

                    print(f'Image {ct} (fail={ct_fail}): SUCCESS {path.name}; stars={points.shape[1]}; dx={transform[0][2]:.1f}; dy={transform[1][2]:.1f}')
                    t = transform[0:2]
                    registered = cv2.warpAffine(img_orig_shift, t, (img_orig_shift.shape[1], img_orig_shift.shape[0]))            
                    cv2.imwrite(f'./output/registered_{self.__detector.name}_{ct}.jpg', registered)

                    break # out of while
                except MatchException as exc:
                    retry += 1
                    if retry==1:
                        print(f'Image {ct} (fail={ct_fail}): {exc} (retrying with brute force matcher)')
                        # Try again, use offset from brute force matcher
                        img_proc = improcess(img_orig, self.__preprocessor)
                        points = self.__detect_stars(img_proc)
                        transform_approx = matcher_bf.match(ref_points, points)
                        offset_ref_image = transform_approx.T[2,:2]
                    elif retry==2:
                        print(f'Image {ct} (fail={ct_fail}): {exc} (retrying with zero offset)')
                        # Try again, use offset 0, 0
                        offset_ref_image = np.array([0.0, 0.0])
                    else:
                        # Give up; proceed with next image
                        print(f'Image {ct} (fail={ct_fail}): {exc} (giving up)')
                        ct_fail += 1
                        break
                finally:
                    self.__show_anot_images(self.__ref_file.name, img_ref_proc, ref_points, path.name, img_proc, points, transform)


    def execute(self, param):
        self.__check_parameter(param)

        if self.__param.star_detector == StarDetectionMethod.Blob:
            preprocess = MacroProcessor()
            preprocess.add(NccProcessor('./images/pattern2.png', retain_size=True))
            preprocess.add(MedianProcessor(11))
    #       preprocess.add(AdaptiveGuaussianThresholdProcessor())
            preprocess.add(NormalizeProcessor())
            self.__preprocessor = preprocess

            self.__detector = BlobDetector()
        elif self.__param.star_detector == StarDetectionMethod.Ncc:
            preprocess = MacroProcessor()
            preprocess.add(MedianProcessor(11))
            self.__preprocessor = preprocess

            self.__detector = TemplateDetector(threshold = 0.1, max_num = 300)
    #        self.__detector.load('./images/star.png')
            self.__detector.load('./images/pattern4.png')  

        self.__align_stack()