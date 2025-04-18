import numpy as np
import pathlib
import rawpy
import cv2

from typing import Union
from ocvl.processor.processor_base import *


def shift_image(image, offset):
    M = np.float32([[1, 0, offset[0]],
	                [0, 1, offset[1]]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))            
    return shifted
	
# Read an image in jpeg or raw format
def imread(file : str, processor : Union[ProcessorBase, list] | None = None):
	ext = pathlib.Path(file).suffix
	if ext.lower()=='.cr2':
		image : np.ndarray = rawpy.imread(file).postprocess() 

#		image : np.array = rawpy.imread(file).postprocess(output_bps=16) 
#		image = np.float32(image) # image.astype(np.float32)
#		image = image / 65535.0
	else:
		image : np.ndarray = cv2.imread(file)

	original_image = image.copy()

	if type(processor) is list:
		for p in processor:
			image = p.process(image)
	elif type(processor) is ProcessorBase:
		image = processor.process(image)
	elif processor is None:
		pass
	
	return image, original_image


def improcess(image : np.ndarray, processor : Union[ProcessorBase, list] | None = None):
	if image is None:
		raise RuntimeError('improcess: image must not be None!')

	if processor is None:
		raise RuntimeError('improcess: processor must not be None!')

	processed_image = image.copy()

	if type(processor) is list:
		for p in processor:
			processed_image = p.process(processed_image)
	elif type(processor) is ProcessorBase:
		processed_image = processor.process(processed_image)
	elif processor is None:
		pass
	
	return processed_image

# Nonmax Suppresssion algorithm (Malisiewicz et al.)
# https://pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes	
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")