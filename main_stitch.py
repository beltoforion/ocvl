#from ocvl.detectors.template_detector import *
#from ocvl.detectors.keypoint_detector import *
#from ocvl.detectors.blob_detector import *

#from ocvl.processor.scale_processor import *
#from ocvl.processor.macro_processor import *
#from ocvl.processor.ncc_processor import *
#from ocvl.processor.agauss_thresh_processor import *
#from ocvl.processor.greyscale_processor import *
#from ocvl.processor.median_processor import *
#from ocvl.processor.normalize_processor import *

#from ocvl.matcher.icp_matcher import *
#from ocvl.matcher.brute_force_matcher import *

from ocvl.helper.opencv_helper import *
from pathlib import Path


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
        if len(stitched.shape)==2:
            stitched = cv2.cvtColor(stitched, cv2.COLOR_GRAY2BGR)

        normalizedImg = np.zeros((w, h))
        normalizedImg = cv2.normalize(stitched,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
        cv2.imshow("Stitched; Normalized", normalizedImg)
        cv2.waitKey(0)

    else:
        print(f'Stitching failed ({status})')


if __name__ == "__main__":
    path = Path('./images/pano1')
    stitch(path.glob('**/*.jpg'))

    cv2.destroyAllWindows()