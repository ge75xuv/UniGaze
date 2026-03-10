import os, argparse
import os.path as osp
import cv2
import h5py
import numpy as np
from tqdm import tqdm
from glob import glob
from gazelib_interface import *


def grid_images(image_list, grid_shape):
	if len(image_list) != grid_shape[0] * grid_shape[1]:
		raise ValueError("Grid shape incompatible with number of images")
	rows = [image_list[i:i+grid_shape[1]] for i in range(0, len(image_list), grid_shape[1])]
	rows = [cv2.hconcat(row) for row in rows]
	grid = cv2.vconcat(rows)
	return grid





if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", type=str, help='directory of mpiifacegaze raw data')
	config, _ = parser.parse_known_args()
	

	data_dir = config.data_dir
	sample_dir = os.path.join(data_dir, 'read_samples')
	os.makedirs(sample_dir, exist_ok=True)

	subjects = [ osp.basename(f) for f in sorted(glob(osp.join(data_dir, '*.h5')) )  ]
	for subject in subjects[:]:
		print('processing: ', subject)
		data_path = osp.join(data_dir, subject)
	
		with h5py.File(data_path, 'r') as f:
			# print(list(f.keys()))
			number_images = f['face_patch'].shape[0]
			print('Number of images: ', number_images)
			images_pool = []
			# for i in tqdm(range(  0, number_images )):
			# 	image = f['face_patch'][i]
			# 	head_pose = f['face_head_pose'][i]
			# 	gaze = f['face_gaze'][i]
			# 	image = draw_gaze(image, gaze, thickness=4)
			# 	images_pool.append(image)
			# 	if len(images_pool) == 32:
			# 		grid_shape = (4, 8)
			# 		grid = grid_images(images_pool, grid_shape)
			# 		cv2.imwrite( sample_dir + f'/grid_{i}.jpg', grid)
			# 		images_pool = []