import cv2
import numpy as np
import matplotlib.pyplot as plt
 
# def CannyThreshold(lowThreshold):
#     detected_edges = cv2.Canny(img2.copy(),lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
#     dst = cv2.bitwise_and(img,img,mask = detected_edges)  # just add some colours to edges from original image.
#     cv2.imshow('canny demo',dst)
 
# lowThreshold = 0
# max_lowThreshold = 100
# ratio = 3
# kernel_size = 7
 
# img = cv2.imread('E:\wxy-git-document-layout-analysis-master\GC\BEC\GC_BEC_05.jpg')
# img = cv2.resize(img, (1000, 1300))

# img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
# img2 = cv2.pyrMeanShiftFiltering(img, 25, 20);  #meanshift平滑,保护边缘效果好
# cv2.namedWindow('pyrMeanShiftFiltering')
# cv2.resizeWindow("pyrMeanShiftFiltering", 640, 480)
# cv2.imshow("pyrMeanShiftFiltering", img2)
# cv2.waitKey(0)

# # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
# cv2.namedWindow('canny demo')
# cv2.resizeWindow("canny demo", 640, 480)
# cv2.createTrackbar('Min threshold','canny demo',lowThreshold, max_lowThreshold, CannyThreshold)
 
# CannyThreshold(0)  # initialization
# if cv2.waitKey(0) == 27:
#     cv2.destroyAllWindows()



def unevenLightCompensate(img, blockSize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    average = np.mean(gray)

    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))

    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]

            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver

    blockImage = blockImage - average
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst = dst.astype(np.uint8)
    dst = cv2.GaussianBlur(dst, (5, 5), 0)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    return dst


def showMat(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(name, 500, 400)
    cv2.imshow(name, img)
    cv2.waitKey(0)


def sobel(sbimg):
    x = cv2.Sobel(sbimg, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(sbimg, cv2.CV_16S, 0, 1)
    # cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
    # 可选参数alpha是伸缩系数，beta是加到结果上的一个值，结果返回uint类型的图像
    Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
    Scale_absY = cv2.convertScaleAbs(y)
    result = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
    showMat('sobel', result)
    #ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    rgb = cv2.cvtColor(result.copy(), cv2.COLOR_HSV2RGB)
    gray = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2GRAY)
    showMat('gray', gray)

    ret, thresh = cv2.threshold(gray, 3, 255, cv2.THRESH_BINARY)
    showMat('thresh', thresh)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
    # thresh = cv2.erode(thresh, kernel3)
    # thresh = cv2.dilate(thresh, kernel3)
    thresh = cv2.medianBlur(thresh,3)
    showMat('bi', thresh)
    (sbcontours, sbhier) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cpImg = img.copy()
    srcArea = img.shape[0]*img.shape[1]
    for i, c in enumerate(sbcontours):
        if cv2.contourArea(c) < srcArea*0.1:
            continue
        cv2.drawContours(cpImg,sbcontours,i,(0,255,0),2)
    showMat('cpImg', cpImg)





img = cv2.imread("E:\wxy-git-document-layout-analysis-master\GC\BEC\GC_BEC_03.jpg")
img = cv2.resize(img, (600, 900))
showMat('img', img)


# img = cv2.medianBlur(img,3)
# showMat("GaussianBlur", img)

# img = cv2.pyrMeanShiftFiltering(img, 25, 20);  #meanshift平滑,保护边缘效果好
# showMat('pyrMeanShiftFiltering0', img)

hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
showMat('hsv', hsv)
# h, s, v = cv2.split(hsv)

msf = cv2.pyrMeanShiftFiltering(hsv, 25, 20);  #meanshift平滑,保护边缘效果好
showMat('pyrMeanShiftFiltering', msf)

# sobel(msf)

canny = cv2.Canny(msf, 40, 120, 9)
cv2.namedWindow('Canny')
cv2.resizeWindow("Canny", 640, 480)
cv2.imshow('Canny', canny)
cv2.waitKey(0)

kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
canny = cv2.dilate(canny, kernel5)
# canny = cv2.erode(canny, kernel3)

canny = cv2.medianBlur(canny,3)
showMat('dilate', canny)

a = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(a))

(_, contours, hier) = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
areas = []
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    areas.append(area)
    print(area)
print(f'contour count: {len(contours)}')

srcArea = img.shape[0]*img.shape[1]
for i, c in enumerate(contours):
    if cv2.contourArea(c) < srcArea*0.1:
        continue
    cv2.drawContours(img,contours,i,(0,255,0),2)

# cv2.drawContours(img,contours,-1,(0,255,0),2)
showMat('src', img)

# pltnp.array(areas)

cv2.destroyAllWindows()
