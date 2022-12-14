import sys

from detectors.template_detector import *
from detectors.keypoint_detector import *
from detectors.blob_detector import *

from processor.scale_processor import *
from processor.macro_processor import *
from processor.ncc_processor import *
from processor.agauss_thresh_processor import *
from processor.greyscale_processor import *
from processor.median_processor import *
from processor.normalize_processor import *

from matcher.icp_matcher import *
from matcher.brute_force_matcher import *

from helper.opencv_helper import *
from pathlib import Path

from StarStacker import *


def template_detect():
    # onliy normalized methods are supported:
    # cv.TM_CCOEFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_SQDIFF_NORMED
    pat = TemplateDetector(cv2.TM_CCORR_NORMED)
    pat.load('./images/pattern1.png')
    pat.threshold = 0.9

    image, _ = cv2.imread('./images/2b6bba87dc8786be.jpg')
    h, w = image.shape[:2]
    scale = 700/w

    boxes = pat.search(image)

    num = boxes.shape[0]
    print(f'Patterns found: {num}')

    for i in range(num):
        x1, y1, x2, y2, score = boxes[i]
        image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2 )
        print(f'box {i}: {x1}, {y1}, {x2}, {y2}, {score}')

    buf = cv2.resize(image, (int(scale * w), int(scale * h)))
    cv2.imshow('found', buf)
    cv2.waitKey(0)


def keypoint_detect():
    # onliy normalized methods are supported:
    # cv.TM_CCOEFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_SQDIFF_NORMED
#    pat = TemplateDetector(cv2.TM_CCORR_NORMED)
    pat = KeypointDetector()
    pat.load('./images/pattern2.png')
    pat.threshold = 0.4

    image, _ = imread('./images/stack/IMG_8018.CR2')

    h, w = image.shape[:2]
    scale = 1500/w

    boxes = pat.search(image)
    if not boxes is None:
        num = boxes.shape[0]
        print(f'Patterns found: {num}')

        for i in range(num):
            x1, y1, x2, y2, score = boxes[i]
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2 )
            print(f'box {i}: {x1}, {y1}, {x2}, {y2}, {score}')

    buf = cv2.resize(image, (int(scale * w), int(scale * h)))
    cv2.imshow('found', buf)
    cv2.waitKey(0)


def stitch(pathlist):
    images = []
    for path in pathlist:
        image, _ = imread(str(path))
        images.append(image)

    stitcher = cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(images)
    if status == 0:
        # write the output stitched image to disk
        cv2.imwrite("./stitched.jpg", stitched)

	    # display the output stitched image to our screen
        cv2.imshow("Stitched", stitched)
        cv2.waitKey(0)
        
        h, w = stitched.shape[:2]
        if len(stitch.shape)==2:
            stitched = cv2.cvtColor(stitched, cv2.COLOR_GRAY2BGR)

        normalizedImg = np.zeros((w, h))
        normalizedImg = cv2.normalize(stitched,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
        cv2.imshow("Stitched; Normalized", normalizedImg)
        cv2.waitKey(0)

    else:
        print(f'Stitching failed ({status})')


def align_stars():
    param = StarStackerParameter()
    param.star_detector = StarDetectionMethod.Ncc
    param.path = './images/stack_untracked'
    param.ref_image = 'IMG_8018.CR2'
    param.path = './images/stack_untracked2'
    param.ref_image = 'IMG_9191.CR2'
    stacker = StarStacker()
    stacker.execute(param)


def stitch_images():
    path = Path('./images/pano1')
    stitch(path.glob('**/*.jpg'))


if __name__ == "__main__":
    align_stars()

    #keypoint_detect()
    #template_detect()
    cv2.destroyAllWindows()