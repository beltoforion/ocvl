from processor.processor_base import ProcessorBase
import numpy as np
import cv2


class NccProcessor(ProcessorBase):
    def __init__(self, pattern_file, retain_size = True):
        super(NccProcessor, self).__init__("NccProcessor")      
        self._pattern = cv2.imread(pattern_file)
        self._retain_size = retain_size

    def process(self, image : np.array) -> np.array:
        res = cv2.matchTemplate(image, self._pattern, cv2.TM_CCORR_NORMED)
        res = res * 255
        res = cv2.normalize(res.astype(np.uint8), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) 

        if self._retain_size:
            h, w = self._pattern.shape[:2]
            h_match, w_match = res.shape[:2]
            h_orig, w_orig = image.shape[:2]

            h_2 = int(h/2)
            w_2 = int(w/2)

            resized_image = np.zeros((h_orig, w_orig, len(image.shape)), dtype = "uint8")
            resized_image[h_2:h_match + h_2, w_2:w_match + w_2, 0] = res
            resized_image[h_2:h_match + h_2, w_2:w_match + w_2, 1] = res
            resized_image[h_2:h_match + h_2, w_2:w_match + w_2, 2] = res            
#            resized_image[h-1:h_orig, w-1:w_orig, 0] = res
#            resized_image[h-1:h_orig, w-1:w_orig, 1] = res
#            resized_image[h-1:h_orig, w-1:w_orig, 2] = res            
            return resized_image
        else:    
            return res