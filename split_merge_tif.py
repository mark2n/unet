'''
split 30 single images from an array of images : train-volume.tif label-volume.tif test-volume.tif
'''
from libtiff import TIFF3D,TIFF
dirtype = ("train","label","test")

import os, os.path

def split_img():

	'''
	split a tif volume into single tif
	'''
	path = '/home/surpath/Projects/test/data/'
	for t in dirtype:
		imgdir = TIFF3D.open(t + "-volume.tif")
		imgarr = imgdir.read_image()
		for i in range(imgarr.shape[0]):
			imgname = path + t + "/" + str(i) + ".tif"
			try:
				img = TIFF.open(imgname,'w')
				img.write_image(imgarr[i])
			except:
				if not os.path.isdir(path + t + "/"):
					os.mkdir(path + t + "/")
				with open(imgname, 'w') as f: # create a new file if not exist
					img = TIFF.open(imgname,'w')
					img.write_image(imgarr[i])

def pop_img(pop_nr):
	path = '/home/surpath/Projects/unet/data/'
	for t in dirtype:
		imgdir = TIFF3D.open(t + "-volume.tif")
		imgarr = imgdir.read_image()

		imgdir_new = TIFF3D.open(path + t + "-volume_n.tif", "w")
		imgdir_new.write_image(imgarr[:-pop_nr])
		# try:
		# 	imgdir_new = TIFF3D.open(path + t + "-volume_n.tif")
		# 	imgdir_new.write_image(imgarr[:-pop_nr])
		# except:
		# 	with open(path + t + "-volume_n.tif", 'w') as f:
		# 		imgdir_new = TIFF3D.open(path + t + "-volume_n.tif")
		# 		imgdir_new.write_image(imgarr[:-pop_nr])

		

def merge_img():
	
	'''
	merge single tif into a tif volume
	'''
	path = '/home/surpath/Projects/unet/data/'
	imgdir = TIFF3D.open("test_mask_volume_server2.tif",'w')
	imgarr = []
	for i in range(30):
		img = TIFF.open(path + str(i) + ".tif")
		imgarr.append(img.read_image())
	imgdir.write_image(imgarr)

if __name__ == "__main__":

	# merge_img()
	split_img()
	# pop_img(2)


