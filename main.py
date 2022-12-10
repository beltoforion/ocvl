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

from helper.opencv_helper import *
from pathlib import Path
import matplotlib.pyplot as plt


def template_detect():
    # onliy normalized methods are supported:
    # cv.TM_CCOEFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_SQDIFF_NORMED
    pat = TemplateDetector(cv2.TM_CCORR_NORMED)
    pat.load('./images/pattern1.png')
    pat.threshold = 0.9

    image : np.array = cv2.imread('./images/2b6bba87dc8786be.jpg')
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

    image = imread('./images/stack/IMG_8018.CR2')

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


def del_miss(indeces, dist, max_dist, th_rate = 0.8):
    th_dist = max_dist * th_rate
    return np.where(dist.T[0] < th_dist)
#    return np.array([indeces[0][np.where(dist.T[0] < th_dist)]])


def is_converge(Tr, scale):
    delta_angle = 0.0001
    delta_scale = scale * 0.0001
    
    min_cos = 1 - delta_angle
    max_cos = 1 + delta_angle
    min_sin = -delta_angle
    max_sin = delta_angle
    min_move = -delta_scale
    max_move = delta_scale
    
    return min_cos < Tr[0, 0] and Tr[0, 0] < max_cos and \
           min_cos < Tr[1, 1] and Tr[1, 1] < max_cos and \
           min_sin < -Tr[1, 0] and -Tr[1, 0] < max_sin and \
           min_sin < Tr[0, 1] and Tr[0, 1] < max_sin and \
           min_move < Tr[0, 2] and Tr[0, 2] < max_move and \
           min_move < Tr[1, 2] and Tr[1, 2] < max_move


def icp(d1, d2, max_iterate = 100):
    src = np.array([d1.T], copy=True).astype(np.float32)
    dst = np.array([d2.T], copy=True).astype(np.float32)
    
    knn = cv2.ml.KNearest_create()
    responses = np.array(range(len(d1[0]))).astype(np.float32)
    knn.train(src[0], cv2.ml.ROW_SAMPLE, responses)
        
    trans = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])

    max_dist = 10000000000000
    scale_x = np.max(d1[0]) - np.min(d1[0])
    scale_y = np.max(d1[1]) - np.min(d1[1])
    scale = max(scale_x, scale_y)

    for i in range(max_iterate):
        ret, results, neighbours, dist = knn.findNearest(dst[0], 1)
        
        indeces = results.astype(np.int32).T     
        keep_idx = del_miss(indeces, dist, max_dist)  
        indeces = np.array([indeces[0][keep_idx]])       
        dst = dst[0, keep_idx]

        T, T2 = cv2.estimateAffinePartial2D(dst[0], src[0, indeces],  True)
        if T is None:
            print(f'{max_dist:0.1f};{dst.shape[1]}_', end = '')
            break

        max_dist = np.max(dist)
        dst = cv2.transform(dst, T)
        trans = np.dot(np.vstack((T,[0,0,1])), trans)

#        plt.plot(d1[0], d1[1], marker='o', linestyle = 'None')
#        xxx = np.array([d2.T], copy=True).astype(np.float32)
#        xxx = cv2.transform(xxx, Tr[0:2])
#        plt.plot(xxx[0].T[0], xxx[0].T[1], marker='+', linestyle = 'None')
#        plt.show()

        if (is_converge(T, scale)):
            print(f'{max_dist:0.1f};{dst.shape[1]}_', end = '')
            return trans

        print('.', end = '')

    return None

def build_comparison(orig_image, stars, size = 1500):
    anot_image = orig_image.copy()

    for pos in stars:
        x, y, w, h, score, clid = pos
        cv2.rectangle(anot_image, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0,255,0), 3)

    overview = np.concatenate((orig_image, anot_image), axis=1)

    h, w = overview.shape[:2]
    scale = size / w
    overview = cv2.resize(overview, (int(scale*w), int(scale*h)))

    if len(overview.shape)==2:
        overview = cv2.cvtColor(overview, cv2.COLOR_GRAY2BGR)    

    return overview

def shift_image(image, offset):
    M = np.float32([[1, 0, offset[0]],
	                [0, 1, offset[1]]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))            
    return shifted

def align_stack(pathlist, ref_file : pathlib.Path, detector : DetectorBase, process : ProcessorBase, show_original):
    ref_image = imread(str(ref_file), process)
    h, w = ref_image.shape[:2]

    # Copy reference image twice and convert to color if not already in color
    if show_original:
        orig_image = imread(str(ref_file))
    else:
        orig_image = ref_image

    anot_image = orig_image.copy()   # for detection boxes

    if len(anot_image.shape)==2:
        anot_image = cv2.cvtColor(anot_image, cv2.COLOR_GRAY2BGR)        

    if len(orig_image.shape)==2:
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_GRAY2BGR)        

    stars = detector.search(ref_image)
    ref_points = np.array([(s[0], s[1]) for s in stars]).T

    print(f'Number of stars detected by {detector.name}: {len(stars)}')

    overview_ref = build_comparison(orig_image, stars)
    
    plt.ion()

    ct = 0
    ct_fail = 0
    transform = None
    last_offset = np.array([0.0, 0.0])
    for path in pathlist:
        if path.name == ref_file.name:
            continue

        image_unprocessed = imread(str(path))
        image = imread(str(path), process)

        # Shift image by last offset
        if not last_offset is None:
            image = shift_image(image, last_offset)
            image_unprocessed = shift_image(image_unprocessed, last_offset)

        ct += 1

        stars = detector.search(image)

        overview = build_comparison(image, stars)
        overview = np.concatenate((overview_ref, overview), axis=0)
        cv2.imshow('Detection Result', overview)
        cv2.waitKey(100)

        points = np.array([(s[0], s[1]) for s in stars]).T

        print(f'Image {ct} (fail={ct_fail}): {ref_file.name} - {path.name}: Starting ICP', end = '')
        transform = icp(ref_points, points)
        if transform is None:
            last_offset = np.array([0.0, 0.0])
            print('FAILED')                    
            ct_fail += 1
            continue
        else:
            last_offset += transform[:2, 2] #(transform[0][2], transform[1][2])
            print(f'SUCCESS (dx={transform[0][2]:.1f}, dy={transform[1][2]:.1f})')

            t = transform[0:2]
            registered = cv2.warpAffine(image_unprocessed, t, (image_unprocessed.shape[1], image_unprocessed.shape[0]))            
            cv2.imwrite(f'./output/registered_{detector.name}_{ct}.jpg', registered)
            
            plt.clf()
            plt.plot(ref_points[0], ref_points[1], marker='o', linestyle = 'None')
            dst = np.array([points.T], copy=True).astype(np.float32)
            dst = cv2.transform(dst, t)
            plt.title(f'{ct}: {ref_file.name}-{path.name} ({transform[0][2]:.0f}x{transform[1][2]:.0f})')
            plt.plot(dst[0].T[0], dst[0].T[1], marker='+', linestyle = 'None')
            plt.draw()
            plt.pause(0.0001)
            plt.savefig(f'./output/match_{ct}.png')


def stitch(pathlist):
    images = []
    for path in pathlist:
        image = imread(str(path))
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
        process.add(MedianProcessor(5))

        detector = TemplateDetector(threshold = 0.1, max_num = 800)
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
    align_stack(pathlist, ref_image, detector, process, show_original)


def stitch_images():
    path = Path('./images/pano1')
    stitch(path.glob('**/*.jpg'))


if __name__ == "__main__":
    align_stars()

    #keypoint_detect()
    #template_detect()
    cv2.destroyAllWindows()