from detectors.template_detector import *

def main():
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

if __name__ == "__main__":
    main()