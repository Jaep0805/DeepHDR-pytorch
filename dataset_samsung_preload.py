from torch.utils.data import Dataset, DataLoader
from utils_tf import *
from imageio import imread
import numpy as np
import random

def ReadLabel_Samsung(fileName):
    #label = imageio.imread(os.path.join(fileName, 'HDRImg.hdr'), 'hdr')
    #label = label[:, :, [2, 1, 0]]  ##cv2
    label = cv2.imread(os.path.join(fileName, 'ref_hdr_aligned_linear.hdr'), -1)
    label = label.astype(np.float32)
    return label

def random_augment(img, r_idx, f_idx):
	if r_idx == 1:
		img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
	elif r_idx == 2:
		img = cv2.rotate(img, cv2.ROTATE_180)
	elif r_idx == 3:
		img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
	if f_idx > 0:
		img = img[:, ::-1, :]
	return img

class Samsung_Dataset_test(Dataset):
	def __init__(self, scene_directory):
		list = os.listdir(scene_directory)
		self.image_list = []
		self.num = 0
		for scene in range(len(list)):
			expo_path=os.path.join(scene_directory, list[scene], 'input_exp.txt')
			file_path=list_all_files_sorted(os.path.join(scene_directory, list[scene]), 'input_*.tif')
			label_path=os.path.join(scene_directory, list[scene])
			self.image_list += [[expo_path, file_path, label_path]]
			self.num = self.num + 1
		self.expoTimes_list = []
		self.ldr_images_list = []
		self.label_list = []
		for i in self.image_list:
			self.expoTimes_list.append(ReadExpoTimes(i[0]))
			img = ReadImages(i[1])
			self.ldr_images_list.append(img)
			self.label_list.append(ReadLabel_Samsung(i[2]))


	def __getitem__(self, idx):
		# Read Expo times in scene
		#expoTimes = ReadExpoTimes(self.image_list[idx][0]) #exposure time
		#print(expoTimes)
		# Read Image in scene
		#imgs = ReadImages(self.image_list[idx][1])
		# Read label
		#label_raw = ReadLabel_Samsung(self.image_list[idx][2])
		expoTimes = self.expoTimes_list[idx]
		imgs = self.ldr_images_list[idx]
		label_raw = self.label_list[idx]
		# inputs-process
		pre_img0 = LDR_to_HDR(imgs[0], expoTimes[0], 2.2)
		pre_img1 = LDR_to_HDR(imgs[1], expoTimes[1], 2.2)
		pre_img2 = LDR_to_HDR(imgs[2], expoTimes[2], 2.2)

		# label-process
		label = label_raw
		#label = range_compressor(label_raw)
		# argument
		crop_size = 256
		H, W, _ = imgs[0].shape

		im1 = imgs[0].astype(np.float32)
		im2 = imgs[1].astype(np.float32)
		im3 = imgs[2].astype(np.float32)
		im4 = label.astype(np.float32)

		im1 = im1.transpose(2, 0, 1)
		im2 = im2.transpose(2, 0, 1)
		im3 = im3.transpose(2, 0, 1)
		im4 = im4.transpose(2, 0, 1)
		# print(im1.shape)
		im1 = im1[::-1,...].copy()
		im2 = im2[::-1,...].copy()
		im3 = im3[::-1,...].copy()
		im4 = im4[::-1,...].copy()
  
  
		im1_hdr = LDR_to_HDR(im1, expoTimes[0], 2.2)
		im2_hdr = LDR_to_HDR(im2, expoTimes[1], 2.2)
		im3_hdr = LDR_to_HDR(im3, expoTimes[2], 2.2)
  
		#normalize to -1~1
		im1 = (im1 *2.0)-1.0
		im2 = (im2 *2.0)-1.0
		im3 = (im3 *2.0)-1.0
		im4 = (im4 *2.0)-1.0
  
		im1_hdr = (im1_hdr *2.0)-1.0
		im2_hdr = (im2_hdr *2.0)-1.0
		im3_hdr = (im3_hdr *2.0)-1.0
		
  
		label_raw = label_raw.astype(np.float32)#.transpose(2, 0, 1)
		label_raw = label_raw.transpose(2, 0, 1)
		label_raw = label_raw[::-1,...].copy()
		label_raw = torch.from_numpy(label_raw)
		
		gtLl = torch.clamp(label_raw*expoTimes[0], 0.0, 1.0) #hdr to ldr
		gtLh = torch.clamp(label_raw*expoTimes[2], 0.0, 1.0)
		gtLl = gtLl ** (1./2.2)
		# gt_LDR_mid_patch = np.power(gt_LDR_mid_patch, 1./gamma)
		gtLh = gtLh ** (1./2.2)


		ref_HDR = im4
		in_LDR = np.concatenate((im1, im2, im3), axis = 0)
		in_HDR = np.concatenate((im1_hdr, im2_hdr, im3_hdr), axis = 0)

	
		#sample = {'rawinput0' : im1, 'rawinput1' : im2,'rawinput2' : im3,
		#'input0': im1_hdr, 'input1': im2_hdr, 'input2': im3_hdr, 
		#'hdr_low0': gtLl, 'hdr_low1': im2, 'hdr_low2': gtLh,
		#'label': im4, 'expo0': expoTimes[0], 'expo1': expoTimes[1], 'expo2': expoTimes[2]}
		return in_LDR.copy().astype(np.float32), in_HDR.copy().astype(np.float32), ref_HDR.copy().astype(np.float32)
		# sample = (in_LDR, in_HDR, im4)
		# return sample

	def __len__(self):
		return self.num


class Samsung_Dataset(Dataset):
	def __init__(self, scene_directory):
		list = os.listdir(scene_directory)
		self.image_list = []
		self.num = 0
		for scene in range(len(list)):
			expo_path=os.path.join(scene_directory, list[scene], 'input_exp.txt')
			file_path=list_all_files_sorted(os.path.join(scene_directory, list[scene]), 'input_*.tif')
			label_path=os.path.join(scene_directory, list[scene])
			self.image_list += [[expo_path, file_path, label_path]]
			self.num = self.num + 1
		self.expoTimes_list = []
		self.ldr_images_list = []
		self.label_list = []
		for i in self.image_list:
			self.expoTimes_list.append(ReadExpoTimes(i[0]))
			img = ReadImages(i[1])
			self.ldr_images_list.append(img)
			self.label_list.append(ReadLabel_Samsung(i[2]))
	def __getitem__(self, idx):
		# Read Expo times in scene
		#expoTimes = ReadExpoTimes(self.image_list[idx][0]) #exposure time
		#print(expoTimes)
		# Read Image in scene
		#imgs = ReadImages(self.image_list[idx][1])
		# Read label
		#label_raw = ReadLabel_Samsung(self.image_list[idx][2])
		expoTimes = self.expoTimes_list[idx]
		imgs = self.ldr_images_list[idx]
		label_raw = self.label_list[idx]
		# inputs-process
		pre_img0 = LDR_to_HDR(imgs[0], expoTimes[0], 2.2)
		pre_img1 = LDR_to_HDR(imgs[1], expoTimes[1], 2.2)
		pre_img2 = LDR_to_HDR(imgs[2], expoTimes[2], 2.2)

		# label-process
		#label = range_compressor(label_raw)
		label = label_raw
		# argument
		crop_size = 256
		H, W, _ = imgs[0].shape
		x = np.random.randint(0, H - crop_size - 1)
		y = np.random.randint(0, W - crop_size - 1)

		im1 = imgs[0][x:x + crop_size, y:y + crop_size, :].astype(np.float32)
		im2 = imgs[1][x:x + crop_size, y:y + crop_size, :].astype(np.float32)
		im3 = imgs[2][x:x + crop_size, y:y + crop_size, :].astype(np.float32)
		im4 = label[x:x + crop_size, y:y + crop_size, :].astype(np.float32)

		r_idx = random.randint(0,3)
		f_idx = random.randint(0,1)
		im1 = random_augment(im1, r_idx, f_idx)
		im2 = random_augment(im2, r_idx, f_idx)
		im3 = random_augment(im3, r_idx, f_idx)
		im4 = random_augment(im4, r_idx, f_idx)

		im1 = im1.transpose(2, 0, 1)
		im2 = im2.transpose(2, 0, 1)
		im3 = im3.transpose(2, 0, 1)
		im4 = im4.transpose(2, 0, 1)
		# print(im1.shape)
		im1 = im1[::-1,...].copy()
		im2 = im2[::-1,...].copy()
		im3 = im3[::-1,...].copy()
		im4 = im4[::-1,...].copy()
  

		im1_hdr = LDR_to_HDR(im1, expoTimes[0], 2.2)
		im2_hdr = LDR_to_HDR(im2, expoTimes[1], 2.2)
		im3_hdr = LDR_to_HDR(im3, expoTimes[2], 2.2)
  
		#normalize to -1~1
		im1 = (im1 *2.0)-1.0
		im2 = (im2 *2.0)-1.0
		im3 = (im3 *2.0)-1.0
		im4 = (im4 *2.0)-1.0
  
		im1_hdr = (im1_hdr *2.0)-1.0
		im2_hdr = (im2_hdr *2.0)-1.0
		im3_hdr = (im3_hdr *2.0)-1.0
  
		label_raw = label_raw[x:x + crop_size, y:y + crop_size, :].astype(np.float32)#.transpose(2, 0, 1)
		label_raw = random_augment(label_raw, r_idx, f_idx)
		label_raw = label_raw.transpose(2, 0, 1)
		label_raw = label_raw[::-1,...].copy()
		label_raw = torch.from_numpy(label_raw)
		
		gtLl = torch.clamp(label_raw*expoTimes[0], 0.0, 1.0)
		gtLh = torch.clamp(label_raw*expoTimes[2], 0.0, 1.0)
		gtLl = gtLl ** (1./2.2)
		# gt_LDR_mid_patch = np.power(gt_LDR_mid_patch, 1./gamma)
		gtLh = gtLh ** (1./2.2)
  

		ref_HDR = im4
  
		in_LDR = np.concatenate((im1, im2, im3), axis = 0)
		in_HDR = np.concatenate((im1_hdr, im2_hdr, im3_hdr), axis = 0)
  
		#sample = {'rawinput0' : im1, 'rawinput1' : im2,'rawinput2' : im3,
		#'input0': im1_hdr, 'input1': im2_hdr, 'input2': im3_hdr, 
		#'hdr_low0': gtLl, 'hdr_low1': im2, 'hdr_low2': gtLh,
		#'label': im4, 'expo0': expoTimes[0], 'expo1': expoTimes[1], 'expo2': expoTimes[2]}
		return in_LDR.copy().astype(np.float32), in_HDR.copy().astype(np.float32), ref_HDR.copy().astype(np.float32)

	def __len__(self):
		return self.num


def main():
	train_loaders = torch.utils.data.DataLoader(Samsung_Dataset("/data2/jaep0805/samsungdataset/CVPR2020_NTIRE_Workshop/SynHDR_Dataset2"), batch_size= 4, shuffle=True, num_workers=4)
	for idx, sample in enumerate(train_loaders):
		#print(idx)
		print(sample['rawinput0'][0])
		print(sample['input0'][0])
		#4x3x256x256
		break
	
 	
if __name__ == "__main__":
	main()
