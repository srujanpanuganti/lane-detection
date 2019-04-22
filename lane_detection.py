import numpy as np
import cv2

############  function to return good corners of the bilateral image  ##################
def good_feat(img):

	size_img = img.shape
	#print(size_img)
	s_x = size_img[0]

	########### upper segment ################
	crop_img1 = img[0:s_x, 500:700]
	#cv2.imshow('crop1',crop_img1)

	########### bottom segment   #################
	crop_img2 = img[200:s_x, 200:1200]
	#cv2.imshow('crop2',crop_img2)

	############# defining the world coordinates  #############
	world_coordinates = np.array([[50,0],[250,0],[250,550],[50,550]],dtype='float32')

	########### detecting corners  ##########################
	corners1 = cv2.goodFeaturesToTrack(crop_img1,2,0.01,mask=None,minDistance=150,blockSize=1,k=10)
	corners2 = cv2.goodFeaturesToTrack(crop_img2,2,0.1,mask=None,minDistance=150,blockSize=1,k=10)

	c1 = []
	###############  Draw corners for top segment ##################

	for i in corners1:

		x,y = i.ravel()
		#print(x,y)
		#p = int(x + 400)
		p = int(x + 500)
		q = int(y + 0)
		c1.append([p,q])

		cv2.circle(img,(p,q),10,(255,0,0),-1)

	###############  Draw corners for bottom segment1 ##################
	for i in corners2:

		#c2 = np.empty([1,2])
		x,y = i.ravel()
		#print(x,y)
		p = int(x + 200)
		q = int(y + 200)
		c1.append([p,q])
		cv2.circle(img,(p,q),10,(255,0,0),-1)
	#print(c1)

	ordered_corners = order_point(c1)
	#cv2.imshow('corners',img)
	return ordered_corners,world_coordinates

###################  to order the points as required for calculatig the homography   ######
def order_point(points):

	rect = []
	rect.append(points[0])
	rect.append(points[1])
	rect.append(points[3])
	rect.append(points[2])

	rect_array = np.asarray(rect,dtype='float32')
	#print(rect_array)

	return rect_array

###################  function to correct the distortion in the image #################
def correct_dist(initial_img):
	################  given calibration matrix   #################
	k = [[1.15422732e+03, 0.00000000e+00, 6.71627794e+02], [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
	k = np.array(k)

	################ Distortion Matrix  ########################
	dist = [[-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]]
	dist = np.array(dist)
	img_2 = cv2.undistort(initial_img, k, dist, None, k)

	return img_2

##################  function for the image operations to prepare the image ############
def image_operations(frame):

	########################## image cropping  ##############
	crop_frame = frame[420:720, 40:1280, :]  # To get the region of interest
	cropped_image = correct_dist(crop_frame)
	#print(cropped_image.shape)

	##############  gray scale image #############
	gray = cv2.cvtColor(cropped_image,cv2.COLOR_BGR2GRAY)
	#cv2.imshow('grray image',gray)

	###############  image thresholding ##############
	x, threshold_image = cv2.threshold(gray,180,255,cv2.THRESH_BINARY)
	#cv2.imshow('threshold image', threshold_image)

	################ bilateralfiltering ###########
	b_filter = cv2.bilateralFilter(threshold_image,1,100,100)
	#cv2.imshow('bilateral filtered image',b_filter)

	return b_filter,crop_frame

################# function to fit a polynomial on the laneS ###########################
def polynomial_fitting(warped_image, H, cropped_image):

	#################  defining the number of windows we want to use to make a polynomial interpolation  ############
	number_of_windows = 50   ############## good number of windows gives use a good polynomia fit
	offset = 10
	minimum_pixel = 10

	cropped_height = cropped_image.shape[1]
	cropped_width = cropped_image.shape[0]
	#print(cropped_height,'\n',cropped_width)      #1240,300

	warped_height = warped_image.shape[0]
	#print(warped_height,'\n',warped_width)    #250,600

	######### histogram equalization ##################
	histogram = np.sum(warped_image, axis=0)
	histogram_height = histogram.shape[0]
	#print(histogram_width)
	#cv2.imshow('histogram',histogram)

	######### Window height #############################
	window_height = warped_height//number_of_windows
	#print(window_height)

	######### Window width #############################
	histogram_midpoint = histogram_height//2
	#print(histogram_midpoint)
	left_window_location = np.argmax(histogram[:histogram_midpoint])
	right_window_location = np.argmax(histogram[histogram_midpoint:])+histogram_midpoint
	#print(histogram_midpoint,'\n',left_window_location,'\n',right_window_location)

	#########  active pixel locations  #################
	active_pixels = warped_image.nonzero()
	active_pixels_x = np.array(active_pixels[1])
	active_pixels_y = np.array(active_pixels[0])
	#print(active_pixels,'\n',active_pixels_x,'\n',active_pixels_y)

	left_window_location_temp = left_window_location
	right_window_location_temp = right_window_location

	left_lane_pixels = []
	right_lane_pixels = []

	###############  creating windows on the warped image at the locations where there is a peak observed in Histogram ###########
	for window_number in range(number_of_windows):
		window_y_bottom = warped_height - window_number * window_height
		window_y_top = warped_height - (window_number+1)* window_height
		window_left_x_low  = left_window_location_temp - offset
		window_left_x_high = left_window_location_temp + offset
		window_right_x_low = right_window_location_temp - offset
		window_right_x_high = right_window_location_temp + offset

		#############  selecting the active pixels in the window  ######################
		desired_left_pixels = ((active_pixels_y >= window_y_top ) & (active_pixels_y < window_y_bottom)
							  & (active_pixels_x >= window_left_x_low) & (active_pixels_x < window_left_x_high)).nonzero()[0]
		desired_right_pixels = ((active_pixels_y >= window_y_top ) & (active_pixels_y < window_y_bottom)
								& (active_pixels_x >= window_right_x_low) & (active_pixels_x < window_right_x_high)).nonzero()[0]

		left_lane_pixels.append(desired_left_pixels)
		right_lane_pixels.append(desired_right_pixels)

		if len(desired_left_pixels) > minimum_pixel:
			left_window_location_temp = np.int(np.mean(active_pixels_x[desired_left_pixels]))
		if len(desired_right_pixels) > minimum_pixel:
			right_window_location_temp = np.int(np.mean(active_pixels_x[desired_right_pixels]))

	###############  concatenating all the active pixels  #############
	left_lane_pixels = np.concatenate(left_lane_pixels)
	right_lane_pixels = np.concatenate(right_lane_pixels)

	final_left_pixels_x = active_pixels_x[left_lane_pixels]
	final_left_pixels_y = active_pixels_y[left_lane_pixels]
	final_right_pixels_x = active_pixels_x[right_lane_pixels]
	final_right_pixels_y = active_pixels_y[right_lane_pixels]

	result_image = np.dstack((warped_image,warped_image,warped_image))*255
	window_image = np.zeros_like(result_image)

	result_image[active_pixels_y[left_lane_pixels], active_pixels_x[left_lane_pixels]] = [255,0,0]
	result_image[active_pixels_y[right_lane_pixels], active_pixels_x[right_lane_pixels]] = [0,0,255]
	#cv2.imshow('result', result_image)
	#cv2.waitKey(0)

	##################  fitting the Polynomial using the collected active pixels #############################
	left_polynomial = np.polyfit(final_left_pixels_y, final_left_pixels_x, 2)
	right_polynomial = np.polyfit(final_right_pixels_y, final_right_pixels_x, 2)

	plot_y = np.linspace(0, warped_height-1, warped_height)

	left_polynomial_x = left_polynomial[0]*plot_y**2 + left_polynomial[1]*plot_y + left_polynomial[2]
	right_polynomial_x = right_polynomial[0]* plot_y**2 + right_polynomial[1]*plot_y + right_polynomial[2]


	left_line_window1 = np.array([np.transpose(np.vstack([left_polynomial_x - offset, plot_y]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_polynomial_x + offset, plot_y])))])
	left_line_points = np.hstack((left_line_window1, left_line_window2))

	right_line_window1 = np.array([np.transpose(np.vstack([right_polynomial_x-offset, plot_y]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_polynomial_x+offset,plot_y])))])
	right_line_points = np.hstack((right_line_window1, right_line_window2))


	#################  filling Lane area ########################################
	window_image = np.zeros_like(result_image)
	cv2.fillPoly(window_image, np.int_([left_line_points]), (255,0,255))
	cv2.fillPoly(window_image, np.int_([right_line_points]), (255,0,255))

	final_lane_image = cv2.addWeighted(result_image, 2, window_image, 0.8, 0)

	###################### super imposing the detected lanes on to the image ###############
	warped_final_image = cv2.warpPerspective(final_lane_image, np.linalg.inv(H), (cropped_height,cropped_width))
	super_imposed = cv2.bitwise_or(warped_final_image,cropped_image)
	cv2.imshow('Lane Detected', super_imposed)


	################# Radius of Curvature #######################################
	####### According to the United Ststes Lanes Regulations ###################
	#ym_per_pix = 30/720
	#xm_per_pix = 3.7/700

	ym_per_pix = 3.048/100
	xm_per_pix = 3.7/378


	left_polynomial_world = np.polyfit(plot_y*ym_per_pix, left_polynomial_x*xm_per_pix, 2)
	right_polynomial_world = np.polyfit(plot_y*ym_per_pix, right_polynomial_x*xm_per_pix, 2)

	y_max = np.max(plot_y)

	left_curve_radius = ((1 + (2*left_polynomial_world[0]*y_max*ym_per_pix + left_polynomial_world[1])**2)**1.5) / np.absolute(2*left_polynomial_world[0])
	right_curve_radius = ((1 + (2*right_polynomial_world[0]*y_max*ym_per_pix + right_polynomial_world[1])**2)**1.5) / np.absolute(2*right_polynomial_world[0])


	average_curvature = int((left_curve_radius + right_curve_radius)/2)
	#print('left curve radius = ',left_curve_radius,'right curve radius = ',right_curve_radius,'\n''average curvature = ',average_curvature)

	##############  predicting the turn of the way  ####################
	if left_curve_radius <= 1500:
		print('turining left')
	elif left_curve_radius >1500 and left_curve_radius <= 3500:
		print('moving straight')
	else:
		print('turning right')

################## we are reading this single image to find the homagraphy matrix ##############
frame = cv2.imread('project_video_frames16.png')
filtered_image, cropped_frame = image_operations(frame)

###########Good features to track########
src, dst  = good_feat(filtered_image)
#print(src)
#print(dst)

##############  Homography  #########
H_matrix = cv2.getPerspectiveTransform(src,dst)
#print(H_matrix)

cap = cv2.VideoCapture('project_video.mp4')
#cap = cv2.VideoCapture('challenge_video.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))

while cap.isOpened():
	success, frame = cap.read()
	if success is False:
		break
	initial_image = frame
	# print(initial_image.shape)

	############# performing imag operations  ######################
	b_filter,cropped_frame = image_operations(initial_image)

	#############  image warping  ##################################
	warped_image = cv2.warpPerspective(b_filter,H_matrix,(270,600))
	cv2.imshow('warped image',warped_image)

	#####################  Polynomial Fitting   ####################
	polynomial_fitting(warped_image,H_matrix,cropped_frame)

	if cv2.waitKey(25) & 0xff == 27:  # To get the correct frame rate
		cv2.destroyAllWindows()
		break
cap.release()

