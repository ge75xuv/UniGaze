
import os, argparse
import os.path as osp
import cv2
import h5py
import numpy as np
from tqdm import tqdm
from glob import glob
from gazelib_interface import *



def upsample_subject(data_path, target_count=3000):
    with h5py.File(data_path, 'a') as f:
        keys = list(f.keys())
        current_count = f['face_patch'].shape[0]

        if current_count >= target_count:
            print(f"Skipping {osp.basename(data_path)}: already has {current_count} images.")
            return

        extra_needed = target_count - current_count
        print(f"Current count: {current_count} --> Target count: {target_count}.")
        print(f"Appending {extra_needed} samples to {osp.basename(data_path)}...")


        extra_indices = np.random.choice(current_count, extra_needed, replace=True)
        for idx in tqdm(extra_indices, desc="Appending"):
            to_write = {}
            for key in keys:
                add(to_write, key, f[key][idx])
            to_h5(to_write, data_path)




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", type=str, help='directory of mpiifacegaze raw data')
	config, _ = parser.parse_known_args()
	

	data_dir = config.data_dir
	subjects = [ osp.basename(f) for f in sorted(glob(osp.join(data_dir, '*.h5')) )  ]
	for subject in subjects[:]:
		print('processing: ', subject)
		data_path = osp.join(data_dir, subject)
		upsample_subject(data_path)