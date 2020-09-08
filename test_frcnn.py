from __future__ import division
import os
import cv2
import numpy as np
import math
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
				help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to read the metadata related to the training (generated when training).",
				default="config.pickle")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')

(options, args) = parser.parse_args()

if not options.test_path:   # if filename is not given
	parser.error('Error: path to test data must be specified. Pass --path to command line')


config_output_filename = options.config_filename

with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)

if C.network == 'resnet50':
	import keras_frcnn.resnet as nn
elif C.network == 'vgg':
	import keras_frcnn.vgg as nn

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

img_path = options.test_path
#w = 0
#h = 0

def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
	w = width
	h = height
	#print(width)		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	
#print(w,h)
def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)
def rotate_point(pointX, pointY, originX, originY, angle):
	theta = [0,-22.5, -45, -67.5, 90, 112.5, -135, 22.5]
	#theta = [0, 90, 135, 45, 157.5, 112.5, 67.5, 22.5]
	#print(angle)
	angle = (theta[angle])
	if angle > 90:
		angle = -(angle-90)
	else:
		angle = angle
	print(angle)
	angle = (angle*math.pi)/180
    #print(pointX,originX)
    #print(pointY,originY)
	new_x = math.cos(angle) * (pointX-originX) + math.sin(angle) * (pointY-originY) + originX
	new_y = - math.sin(angle) * (pointX-originX) + math.cos(angle) * (pointY-originY) + originY
    
    #print("new X",new_x)
    #print("new Y",new_y)
	#if(new_x < 0):
	#	new_x = 0
	#elif(new_x>width):
	#	new_x=width
	#elif(new_y<0):
	#	new_y=0
	#elif(new_y>height):
	#	new_y = height
	return int( new_x ), int (new_y)

#Since it is not axis alligned, we can not use open CV rectangle funtion to draw. 
def draw_rotated_rectangle(img, x1y1, x1y2, x2y1, x2y2 ):
    #x1y1 = x1y1-center_point
    #plt.imshow( cv2.polylines(image,[np.array([new_x1y1,new_x2y1,new_x2y2,new_x1y2])],True,(0,255,255), 30) )
	cv2.polylines(img,[np.array([(x1y1),(x2y1),(x2y2),(x1y2)])],True,(0,255,255), 5) 
	cv2.imwrite('D:/{}.png',img)
class_mapping = C.class_mapping
angle_mapping = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7}
if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)

if C.network == 'resnet50':
	num_features = 1024
elif C.network == 'vgg':
	num_features = 512

if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
	input_shape_features = (num_features, None, None)
else:
	input_shape_img = (None, None, 3)
	input_shape_features = (None, None, num_features)


img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping),nb_angle_classes = 8, trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

print('Loading weights from {}'.format(C.model_path))
model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs = []

classes = {}

bbox_threshold = 0.8

visualise = True

for idx, img_name in enumerate(sorted(os.listdir(img_path))):
	if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
		continue
	print(img_name)
	st = time.time()
	filepath = os.path.join(img_path,img_name)

	img = cv2.imread(filepath)
	(height, width, _ )= img.shape
	#print(height)
	#print(width)
	print(img)
	X, ratio = format_img(img, C)
	#print(X)
	print("ratio",ratio)
	if K.image_dim_ordering() == 'tf':
		X = np.transpose(X, (0, 2, 3, 1))

	# get the feature maps and output from the RPN
	[Y1, Y2, F] = model_rpn.predict(X)
	print("feature map", F.shape[0],F.shape[1],F.shape[2],F.shape[3])
	

	R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.5)
	#print(R)
	# convert from (x1,y1,x2,y2) to (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]

	# apply the spatial pyramid pooling to the proposed regions
	bboxes = {}
	probs = {}
	probs_angle = 0

	for jk in range(R.shape[0]//C.num_rois + 1):
		ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
		if ROIs.shape[1] == 0:
			break

		if jk == R.shape[0]//C.num_rois:
			#pad R
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:, :curr_shape[1], :] = ROIs
			ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
			ROIs = ROIs_padded

		[P_cls, P_regr, P_angle] = model_classifier_only.predict([F, ROIs])
		print("predicted......................")
		#print(P_cls)
		#print(P_regr)
		#print(P_angle)
		for ii in range(P_cls.shape[1]):

			if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
				continue

			cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
			print(cls_name)
			angle_name = angle_mapping[np.argmax(P_angle[0,ii,:])]
			#print(angle_name)
			#print(ii)
			#print(P_angle[0,ii,:])
			if cls_name not in bboxes:
				bboxes[cls_name] = []
				probs[cls_name] = []
				#probs_angle[angle_name] = []
			(x, y, w, h) = ROIs[0, ii, :]
			print("ROIS x,y,w,h", x,y,w,h)

			cls_num = np.argmax(P_cls[0, ii, :])
			try:
				(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
				tx /= C.classifier_regr_std[0]
				ty /= C.classifier_regr_std[1]
				tw /= C.classifier_regr_std[2]
				th /= C.classifier_regr_std[3]
				x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
			except:
				pass
			bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
			probs[cls_name].append(np.max(P_cls[0, ii, :]))
			#probs_angle[angle_name].append(np.max(P_angle[0,ii,:]))
			#angle_probability = np.max(P_angle[0,ii,:])
			if probs_angle == 0:
				#probs_angle[angle_name].append(np.max(P_angle[0,ii,:]))
				#print("angle_probability",np.max(P_angle[0,ii,:]))
				#probs_angle = np.max(P_angle[0,ii,:])
				probs_angle = angle_name
				#print(probs_angle)
			#print(bboxes)
			#print(probs)
			#print("probs angle",probs_angle)
	all_dets = []

	for key in bboxes:
		bbox = np.array(bboxes[key])

		new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
		for jk in range(new_boxes.shape[0]):
			(x1, y1, x2, y2) = new_boxes[jk,:]
			print("x1,y1,x2,y2", x1,y1,x2,y2)

			(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
			print("real_x1",real_x1)
			print("real_y1",real_y1)
			print(real_x2)
			print(real_y2)
			xc = int((real_x1 + real_x2)/2)
			yc = int((real_y1 + real_y2)/2)
			#print(yc)
			anglee = probs_angle
			#print("anglesss",anglee)
			x1y1 = rotate_point(real_x1,real_y1, xc, yc, anglee)
			#newx1y1 = (x1,y1)
			#print("newx1y1",x1,y1)
			x2y2 = rotate_point(real_x2,real_y2, xc, yc, anglee)
			x1y2 = rotate_point(real_x1,real_y2, xc, yc, anglee)
			x2y1 = rotate_point(real_x2,real_y1, xc, yc, anglee)
			draw_rotated_rectangle(img, (x1y1), (x1y2), (x2y1), (x2y2))
			#cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

			textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
			all_dets.append((key,100*new_probs[jk]))

			(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
			textOrg = (real_x1, real_y1-0)

			#cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
			#cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
			#cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

	print('Elapsed time = {}'.format(time.time() - st))
	#print(all_dets)
	#cv2.imshow('img', img)
	#cv2.waitKey(0)
	cv2.imwrite('./results_imgs/{}.png'.format(idx),img)
