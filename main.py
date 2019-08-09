# import necessary packages
import numpy as np
import cv2

from matplotlib import pyplot as plt
import numpy as np

import timeit

#processing letter by letter boxing
def process_letter(binary_img,output):	
	# assign the kernel size	
	kernel = np.ones((2,1), np.uint8) # vertical
	# use closing morph operation then erode to narrow the image	
	temp_img = cv2.morphologyEx(binary_img,cv2.MORPH_CLOSE,kernel,iterations=3)
	# temp_img = cv2.erode(binary_img,kernel,iterations=2)		
	letter_img = cv2.erode(temp_img,kernel,iterations=1)
	
	# find contours 
	(contours, _) = cv2.findContours(letter_img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	
	# loop in all the contour areas
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(output,(x-1,y-5),(x+w,y+h),(0,255,0),1)

	return output	


#processing letter by letter boxing
def process_word(binary_img,output):	
	# assign 2 rectangle kernel size 1 vertical and the other will be horizontal	
	kernel = np.ones((2,1), np.uint8)
	kernel2 = np.ones((1,4), np.uint8)
	# use closing morph operation but fewer iterations than the letter then erode to narrow the image	
	temp_img = cv2.morphologyEx(binary_img,cv2.MORPH_CLOSE,kernel,iterations=2)
	#temp_img = cv2.erode(binary_img,kernel,iterations=2)	
	word_img = cv2.dilate(temp_img,kernel2,iterations=1)
	
	(contours, _) = cv2.findContours(word_img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(output,(x-1,y-5),(x+w,y+h),(0,255,0),1)

	return output	

#processing line by line boxing
def process_line(binary_img,output):	
	# assign a rectangle kernel size	1 vertical and the other will be horizontal
	kernel = np.ones((1,5), np.uint8)
	kernel2 = np.ones((2,4), np.uint8)	
	# use closing morph operation but fewer iterations than the letter then erode to narrow the image	
	temp_img = cv2.morphologyEx(binary_img,cv2.MORPH_CLOSE,kernel2,iterations=2)
	#temp_img = cv2.erode(binary_img,kernel,iterations=2)	
	line_img = cv2.dilate(temp_img,kernel,iterations=5)
	
	(contours, _) = cv2.findContours(line_img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(output,(x-1,y-5),(x+w,y+h),(0,255,0),1)

	return output	

#processing par by par boxing
def process_par(binary_img, output, distinguish_img:bool=True, blur_count:int=1):	
	# assign a rectangle kernel size

	for _ in range(blur_count):
		binary_img = cv2.medianBlur(binary_img, 3)

	kernel = np.ones((5,5), 'uint8')
	par_img = cv2.dilate(binary_img, kernel, iterations=3)

	par_shape = par_img.shape

	gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
	shape = gray.shape
		
	(contours, _) = cv2.findContours(par_img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		if distinguish_img:
			if x+w > shape[1] or y+h > shape[0]:  #1505,1109
				print(x+w, ",", y+h)
			else:
				# another way
				# mask = np.zeros(gray.shape[:2], np.uint8)
				# mask[y: y+h, x: x+w] = 255
				# hist = cv2.calcHist([gray], [0], mask, [256], [0, 256])
				
				hist = cv2.calcHist([gray[y: y+h, x: x+w]], [0], None, [256], [0, 256])
				count = 0
				for item in hist[30:240]:
					count += item[0]
				for item in hist[:30]:
					count -= item[0]
				for item in hist[-30:]:
					count -= item[0]

				count /= h*w

				if count > 0.1:
					cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
					cv2.putText(output, str(count), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 1)
				elif count != -1:
					cv2.rectangle(output,(x,y),(x+w,y+h),(255,0,0),2)
					cv2.putText(output, str(count), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0), 1)

		else:
			cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),1)
	return output	

#processing margin with paragraph boxing
def process_margin(binary_img,output):	
	# assign a rectangle kernel size
	kernel = np.ones((20,5), 'uint8')	
	margin_img = cv2.dilate(binary_img,kernel,iterations=5)
	
	(contours, _) = cv2.findContours(margin_img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),1)

	return output

#"journal1.jpg"
def processImg(file_path):
	if not os.path.exists(file_path):
		raise FileNotFoundError("file not exist!")

	start_time = timeit.default_timer()

	# loading images
	image = cv2.imread(file_path)

	# hardcoded assigning of output images for the 3 input images
	output_letter = image.copy()
	output_word = image.copy()
	output_line = image.copy()
	output_par = image.copy()
	output_margin = image.copy()

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# clean the image using otsu method with the inversed binarized image
	ret,th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

	file_path = os.path.basename(file_path)
	file_path = file_path.split('.')[0]
	letter_path = os.path.join('output_' + file_path, 'letter')
	word_path = os.path.join('output_' + file_path, 'word')
	line_path = os.path.join('output_' + file_path, 'line')
	par_path = os.path.join('output_' + file_path, 'par')
	margin_path = os.path.join('output_' + file_path, 'margin')
	os.makedirs(letter_path)
	os.makedirs(word_path)
	os.makedirs(line_path)
	os.makedirs(par_path)
	os.makedirs(margin_path)

	# processing and writing the output
	output_letter = process_letter(th,output_letter)
	output_word = process_word(th,output_word)
	output_line = process_line(th,output_line)
	# special case for the 5th output because margin with paragraph is just the 4th output with margin
	cv2.imwrite(letter_path + "/output_letter.jpg", output_letter)	
	cv2.imwrite(word_path + "/output_word.jpg", output_word)
	cv2.imwrite(line_path + "/output_line.jpg", output_line)

	# 图像噪声较多时，blur_count可以多一点(3)；需要找出的图片以线条为主时，blur_count要少(0)
	output_par = process_par(th, output_par, distinguish_img=True, blur_count=0) 
	cv2.imwrite(par_path + "/output_par.jpg", output_par)
	output_margin = process_margin(th,output_par)
	cv2.imwrite(margin_path + "/output_margin.jpg", output_margin)

	end_time = timeit.default_timer()
	print('Total time: ', end_time-start_time)


if __name__ == '__main__':
	
	import os
	os.chdir('E:\wxy-git-document-layout-analysis-master\document-layout-analysis')

	processImg('2.jpg')


	# 2.jpg  	    0.018629    855*530
	# 1.jpg  	    0.014034    636*524
	# news1.jpg     0.175249    2500*1909
	# journal1.jpg  0.040054    1109*1505