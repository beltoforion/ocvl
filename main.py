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

from helper.opencv_helper import *
from pathlib import Path


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


def show_anot_images(ref_img_name, orig_image, ref_points, img_name, image, points, transform, windows_size = 1500):
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
    cv2.waitKey(1)


def align_stack(pathlist, ref_file : pathlib.Path, detector : DetectorBase, process : ProcessorBase, show_original):
    ref_image, orig_image = imread(str(ref_file), process)
    if not show_original:
        orig_image = ref_image

    anot_image = orig_image.copy()   # for detection boxes

    if len(anot_image.shape)==2:
        anot_image = cv2.cvtColor(anot_image, cv2.COLOR_GRAY2BGR)        

    if len(orig_image.shape)==2:
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_GRAY2BGR)        

    ref_stars = detector.search(ref_image)
    ref_points = np.array([(s[0], s[1]) for s in ref_stars]).T

    print(f'Number of stars detected by {detector.name}: {len(ref_stars)}')

    ct = 0
    ct_fail = 0
    transform = None
    last_offset = np.array([0.0, 0.0])
    
    matcher = IcpMatcher()
    matcher.max_iterations = 100
    matcher.median_threshold = 10

    for path in pathlist:
        if path.name == ref_file.name:
            continue

        image, image_unprocessed = imread(str(path), process)

        retry = 0
        while retry<2:
            try:
                # Shift image by last offset. This makes the point cloud matching more 
                # likely to succeed. Because images in a series are close to one another
                # and if the last image could be matched the offset is almost correct.
                image = shift_image(image, last_offset)
                image_unprocessed = shift_image(image_unprocessed, last_offset)

                ct += 1

                stars = detector.search(image)
                points = np.array([(s[0], s[1]) for s in stars]).T

                print(f'Image {ct} (fail={ct_fail}): {ref_file.name} - {path.name}: ', end = '')

                try:
                    transform = matcher.match(ref_points, points)
                    last_offset += transform[:2, 2]
                    print(f'SUCCESS (dx={transform[0][2]:.1f}, dy={transform[1][2]:.1f})')

                    t = transform[0:2]
                    registered = cv2.warpAffine(image_unprocessed, t, (image_unprocessed.shape[1], image_unprocessed.shape[0]))            
                    cv2.imwrite(f'./output/registered_{detector.name}_{ct}.jpg', registered)
                finally:
                    show_anot_images(ref_file.name, orig_image, ref_points, path.name, image, points, transform)

                break

            except MatchException as exc:
                print(exc)                    
                retry += 1
                ct_fail += 1
                last_offset = np.array([0.0, 0.0])


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

    method = 0

    if method == 0:
        process = MacroProcessor()
        process.add(NccProcessor('./images/pattern2.png', retain_size=True))
#        process.add(GreyscaleProcessor())
        process.add(MedianProcessor(11))
#        process.add(AdaptiveGuaussianThresholdProcessor())
        process.add(NormalizeProcessor())

        detector = BlobDetector()
        show_original = False

    elif method == 1:
        process = MacroProcessor()
#        process.add(GreyscaleProcessor())
#        process.add(MedianProcessor(5))

        detector = TemplateDetector(threshold = 0.1, max_num = 200)
        detector.load('./images/star.png')
#        detector.load('./images/pattern2.png')  
        show_original = False

    elif method ==2:
        process = MacroProcessor()
        process.add(NccProcessor('./images/pattern2.png', retain_size=True))
#        process.add(GreyscaleProcessor())
        process.add(MedianProcessor(11))
#        process.add(AdaptiveGuaussianThresholdProcessor())
#        process.add(NormalizeProcessor())   

        detector = KeypointDetector()
        show_original = False        

    path = Path('./images/stack_untracked')
    pathlist = path.glob('**/*.CR2')
    ref_image = path / 'IMG_8018.CR2'
#    ref_image = path / 'IMG_9197.CR2'
    align_stack(pathlist, ref_image, detector, process, show_original)


def stitch_images():
    path = Path('./images/pano1')
    stitch(path.glob('**/*.jpg'))


if __name__ == "__main__":
    align_stars()

    #keypoint_detect()
    #template_detect()
    cv2.destroyAllWindows()