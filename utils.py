"""
Author: XIN LI


TODO:
1. add a triangle filter to capture the feature changes in the mapper
"""
# ---------------------------------- open-source libs
from mnist import MNIST
from matplotlib import pyplot as plt
import numpy as np
import random
import math

from scipy.linalg import svd
from numpy import diag
from numpy import dot
from numpy import zeros
from sklearn.linear_model import LinearRegression

############################################# KEY functions

# ------------------------------------------- data pre-processing
def binarization(pixels):
	thres = 100

	coord = []
	dim = pixels.shape

	flat_pixels = pixels.flatten()

	for i in range(len(flat_pixels)):
		if flat_pixels[i] == 0:
			flat_pixels[i] = -1

	pixels = np.array(flat_pixels, dtype='float').reshape(dim)

	for i in range(dim[0]):
		for j in range(dim[1]):
			if pixels[i, j] > thres:
				coord.append([i, j])

	return coord

def prep_data(img):
	img = np.array(img, dtype='float').reshape((28, 28))
	img_bin = binarization(img)
	canvas = np.zeros(shape=(28, 28))
		
	for i in range(len(img_bin)):
		canvas[img_bin[i][0]][img_bin[i][1]] = 1

	return canvas

def get_data(size, isTraining, isRandom):
	mndata = MNIST('dataset')
	images_train, labels_train = mndata.load_training()
	images_test, labels_test = mndata.load_testing()

	imgs = False
	labs = False
	idxes = []

	output_imgs = []
	output_labs = []

	if isTraining:
		imgs = images_train
		labs = labels_train
	else:
		imgs = images_test
		labs = labels_test


	if isRandom:
		l = len(labs) - 1

		for i in range(size):
			idxes.append(random.randint(0, l))

	else:
		for i in range(10):
			labs = np.array(labs)
			idx_list = np.where(labs == i)[0].tolist()
			l = len(idx_list) - 1

			for j in range(int(size / 10)):
				idxes.append(idx_list[random.randint(0, l)])


	for i in idxes:
		img = imgs[i]
		prep_img = prep_data(img)
		output_imgs.append(prep_img)
		output_labs.append(labs[i])

	return output_imgs, output_labs


def get_special_data():
	mndata = MNIST('dataset')
	images_train, labels_train = mndata.load_training()
	images_test, labels_test = mndata.load_testing()

	# ------------------------------ 10
	idxes = [
		[34, 68, 108, 260, 667, 2278, 3661, 3929, 3516, 3434], # 0
		[3, 104, 70, 102, 134, 637, 982, 1696, 2306, 470], # 1
		[16, 25, 28, 122, 180, 252, 347, 390, 844, 938], # 2
		[7, 12, 44, 98, 198, 356, 629, 998, 1185, 1432], # 3
		[58, 127, 131, 142, 850, 1094, 1511, 1814, 2449, 3714], # 4
		[244, 236, 316, 417, 625, 719, 886, 1162, 2143, 2780], # 5
		[62, 186, 256, 488, 486, 1190, 1418, 1784, 2296, 2614], # 6
		[42, 15, 223, 305, 567, 522, 686, 954, 1088, 1412], # 7
		[333, 188, 225, 144, 1419, 1790, 2155, 2672, 3338, 3251], # 8
		[19, 87, 226, 322, 525, 589, 755, 930, 1364, 1674], # 9
	]

	imgs = images_train
	labs = labels_train

	output_imgs = []
	output_labs = []

	for cat in idxes:
		for i in cat:
			img = imgs[i]
			prep_img = prep_data(img)
			output_imgs.append(prep_img)
			output_labs.append(labs[i])

	print(len(output_imgs))
	print(len(output_labs))
	
	return output_imgs, output_labs
# ------------------------------------------- training


############################################# KEY functions










############################################# Testing Area

def rm_same(m1, m2, m3):
	print(m1.shape)
	print(m2.shape)
	print(m3.shape)

	rm1 = np.ones((16, 16), dtype = np.int)
	rm2 = np.ones((16, 16, 16), dtype = np.int)
	rm3 = np.ones((16, 16, 16, 16), dtype = np.int)

	o1 = np.zeros((10, 16, 16), dtype = np.float)
	o2 = np.zeros((10, 16, 16, 16), dtype = np.float)
	o3 = np.zeros((10, 16, 16, 16, 16), dtype = np.float)

	for i in range(10):
		X = m1[i].astype(int)
		rm1 = np.multiply(rm1, X)

	for i in range(10):
		X = m2[i].astype(int)
		rm2 = np.multiply(rm2, X)

	for i in range(10):
		X = m3[i].astype(int)
		rm3 = np.multiply(rm3, X)
	

	# ---------------------------
	rm1 = (rm1[:, :] - 1) * -1
	rm2 = (rm2[:, :, :] - 1) * -1
	rm3 = (rm3[:, :, :, :] - 1) * -1

	for i in range(10):
		X = m1[i]
		o1[i] = np.multiply(rm1, X)

	for i in range(10):
		X = m2[i]
		o2[i] = np.multiply(rm2, X)

	for i in range(10):
		X = m3[i]
		o3[i] = np.multiply(rm3, X)


	return o1, o2, o3


def rm_zeros_idx(m1, m2, m3):
	o1 = np.zeros((10, 16, 16), dtype = np.float)
	o2 = np.zeros((10, 16, 16, 16), dtype = np.float)
	o3 = np.zeros((10, 16, 16, 16, 16), dtype = np.float)

	for i in range(10):
		for a in range(16):
			for b in range(16):
				if m1[i, a, b] != 0 and a not in [0, 15] and b not in [0, 15]:
					o1[i, a, b] = m1[i, a, b]

	for i in range(10):
		for a in range(16):
			for b in range(16):
				for c in range(16):
					if m2[i, a, b, c] != 0 and a not in [0, 15] and b not in [0, 15] and c not in [0, 15]:
						o2[i, a, b, c] = m2[i, a, b, c]

	for i in range(10):
		for a in range(16):
			for b in range(16):
				for c in range(16):
					for d in range(16):
						if m3[i, a, b, c, d] != 0 and a not in [0, 15] and b not in [0, 15] and c not in [0, 15] and d not in [0, 15]:
							o3[i, a, b, c, d] = m3[i, a, b, c, d]

	return o1, o2, o3


def rm_zeros_idx_test(m1, m2, m3):
	o1 = np.zeros((16, 16), dtype = np.float)
	o2 = np.zeros((16, 16, 16), dtype = np.float)
	o3 = np.zeros((16, 16, 16, 16), dtype = np.float)

	for a in range(16):
		for b in range(16):
			if m1[a, b] != 0 and a not in [0, 15] and b not in [0, 15]:
				o1[a, b] = m1[a, b]

	for a in range(16):
		for b in range(16):
			for c in range(16):
				if m2[a, b, c] != 0 and a not in [0, 15] and b not in [0, 15] and c not in [0, 15]:
					o2[a, b, c] = m2[a, b, c]

	for a in range(16):
		for b in range(16):
			for c in range(16):
				for d in range(16):
					if m3[a, b, c, d] != 0 and a not in [0, 15] and b not in [0, 15] and c not in [0, 15] and d not in [0, 15]:
						o3[a, b, c, d] = m3[a, b, c, d]

	return o1, o2, o3





############################################# Testing Area






























def classifier(pixels):
	max_sum = -1
	max_key = -1

	patterns = {
		0: np.array([[-1, 1, -1],[-1, 1, -1], [-1, 1, -1]]),
		45: np.array([[-1, -1, 1],[-1, 1, -1], [1, -1, -1]]),
		90: np.array([[-1, -1, -1],[1, 1, 1], [-1, -1, -1]]),
		135: np.array([[1, -1, -1],[-1, 1, -1], [-1, -1, 1]]),
		360: np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
		-1: np.array([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]),
	}

	for k in patterns.keys():
		try:
			res = np.multiply(patterns[k], pixels).sum()
			#print(str(k) + "-----> " + str(res))
			if res > max_sum:
				max_sum = res
				max_key = k
		except:
			print(patterns[k])
			print(pixels)

	return max_key

def matrix_scan(pixels, size):
	res = []
	dim = pixels.shape
	boundry_deductor = size - 1

	for i in range(dim[0] - boundry_deductor):
		for j in range(dim[1] - boundry_deductor):
			m = i + size
			n = j + size
			sub_matrix = pixels[i:m, j:n]
			res.append(sub_matrix)

	return res

def abstracting(pixels, pen_size):
	res_flat = []
	res_coord = []
	dim = pixels.shape
	boundry_deductor = pen_size - 1

	pixels_with_penalty = pixels.flatten()

	for i in range(len(pixels_with_penalty)):
		if pixels_with_penalty[i] == 0:
			pixels_with_penalty[i] = -1

	pixels_with_penalty = np.array(pixels_with_penalty, dtype='float').reshape(dim)

	#print(pixels_with_penalty)

	for i in range(dim[0] - boundry_deductor):
		for j in range(dim[1] - boundry_deductor):
			m = i + pen_size
			n = j + pen_size
			sub_matrix = pixels_with_penalty[i:m, j:n]
			local_sum = sub_matrix.sum()
			res_flat.append(local_sum)
			res_coord.append([(i, m), (j, n)])

	new_dim = int(math.sqrt(len(res_flat)))
	res_reshaped = np.array(res_flat).reshape((new_dim, new_dim))

	return res_reshaped, res_coord

def is_overlap(area_a, area_b):
	x_gap = area_a[0][1] - area_a[0][0]
	y_gap = area_a[1][1] - area_a[1][0]

	x_distance = abs(area_a[0][1] - area_b[0][1])
	y_distance = abs(area_a[1][1] - area_b[1][1])

	if x_distance > x_gap:
		return False
	elif y_distance > y_gap:
		return False
	else:
		return True

def is_ready_to_paint(area_idx, painted_idx, abstracted_coord):
	for i in range(len(painted_idx)):
		if painted_idx[i] == 1:
			check_area = abstracted_coord[area_idx]
			painted_area = abstracted_coord[i]
			if is_overlap(check_area, painted_area):
				return False

	return True

def get_coord(replica):
	coord = []
	shape = replica.shape
	for x in range(shape[0]):
		for y in range(shape[1]):
			if replica[x][y] == 1:
				coord.append([x, y])

	return np.array(coord)

def get_fake_coord():

	# 4
	coord = [
		[3, 12],
		[5, 11],
		[7, 10],
		[9, 9],
		[11, 8],
		[13, 7],
		[15, 6],
		[15, 10],
		[15, 13],
		[15, 15],
		[15, 17],
		[3, 13],
		[7, 13],
		[11, 13],
		[15, 13],
		[17, 13],
		[21, 13],
	]


	"""
	# 4
	coord = [
		[3, 7],
		[7, 6],
		[10, 6],
		[14, 5],
		[14, 9],
		[14, 13],
		[13, 17],
		[3, 13],
		[7, 13],
		[11, 12],
		[16, 11],
		[21, 10],

	]
	"""

	"""
	# perfec 4
	coord = [
		[3, 7],
		[7, 7],
		[10, 7],
		[13, 7],
		[13, 10],
		[13, 13],
		[13, 15],
		[13, 17],
		[3, 13],
		[7, 13],
		[11, 13],
		[13, 13],
		[17, 13],
		[21, 13],
	]
	"""
	

	"""
	# perfect 2
	coord = [
		[6, 5],
		[4, 7],
		[4, 9],
		[4, 11],
		[5, 13],
		[7, 14],
		[9, 13],
		[11, 12],
		[13, 11],
		[15, 10],
		[17, 9],
		[19, 8],
		[20, 6],
		[21, 5],
		[21, 8],
		[21, 11],
		[21, 14],
	]
	"""


	"""
	# hand 2
	coord = [
		[6, 5],
		[4, 9],
		[5, 13],
		[10, 14],
		[17, 11],
		[20, 6],
		[16, 4],
		[15, 8],
		[19, 15],
		[19, 19],
	]


	"""


	"""
	# hand 2
	coord = [
		[6, 5],
		[4, 7],
		[4, 9],
		[4, 11],
		[5, 13],
		[7, 14],
		[10, 14],
		[13, 13],
		[15, 12],
		[17, 11],
		[18, 10],
		[19, 8],
		[20, 6],
		[18, 4],
		[16, 4],
		[15, 6],
		[15, 8],
		[16, 10],
		[18, 13],
		[19, 15],
		[20, 17],
		[19, 19],
	]
	"""





	"""
	# perfect 7
	coord = [
		[6, 5],
		[6, 7],
		[6, 9],
		[6, 11],
		[6, 13],
		[6, 15],
		[8, 14],
		[10, 13],
		[12, 12],
		[14, 11],
		[16, 10],
		[18, 9],
		[20, 8],
		[22, 8],

	]
	"""



	"""	
	# 7
	coord = [
		[4, 3],
		[3, 5],
		[3, 7],
		[2, 9],
		[5, 9],
		[7, 8],
		[9, 8],
		[11, 7],
		[13, 7],
	]
	"""


	"""
	# 7
	coord = [
		[5, 4],
		[4, 5],
		[3, 7],
		[2, 9],
		[2, 11],
		[3, 12],
		[5, 12],
		[6, 11],
		[8, 10],
		[9, 9],
		[11, 8],
		[13, 7],
	]
	"""



	"""
	coord = [
		[ 0, 13],
		[ 1, 8],
		[ 1, 17],
		[ 3, 4],
		[ 5, 19],
		[ 9, 18],
		[13, 15],
		[17, 13],
		[19,  9],
		[23,  7],
		[24,  3],
		[25, 11],
		[25, 15],
		[25, 19]
	]
	"""
	

	return np.array(coord)

def feature_extraction(pixels, feature):
	res = []

	for i in pixels:
		#if i == 360 or i == feature:
		if i == feature:
			res.append(1)
		else:
			res.append(-1)

	return np.array(res)

# TODO: need optimized this method!!! (there is a potential bug)
def replica(original_pixels, abstracted_pixels, abstracted_coord, pen_size):
	global_max = 0
	pos_cursor = -1
	painted_pos = np.full(abstracted_pixels.shape, 0).flatten().tolist()
	abstracted_pixels_flat = abstracted_pixels.flatten().tolist()
	idx_tracker = {}

	for i in range(len(abstracted_pixels_flat)):
		idx_tracker[i] = abstracted_pixels_flat[i]


	sorted_pixels = sorted(idx_tracker.items(), key=lambda x: x[1], reverse=True)
	#print(type(sorted_pixels))

	for pixel in sorted_pixels:
	#for i in range(3):
		#pixel = sorted_pixels[i]
		#print(painted_pos)
		if global_max == 0:
			global_max += pixel[1]
			painted_pos[pixel[0]] = 1

		elif pixel[1] > 0 and is_ready_to_paint(pixel[0], painted_pos, abstracted_coord):
			global_max += pixel[1]
			painted_pos[pixel[0]] = 1

		#print(painted_pos)


	"""
	# get the first max position
	for i in range(len(abstracted_pixels_flat)):
		local_max = -1
		current_pixel = abstracted_pixels_flat[i]

		print(current_pixel)

		if current_pixel > local_max:
			local_max = current_pixel
			pos_cursor = i

	global_max += local_max
	painted_pos[pos_cursor] = 1







	# find the rest
	for i in range(1, 2):
		for j in range(len(abstracted_pixels_flat)):
			local_max = -1
			current_pixel = abstracted_pixels_flat[j]


			#print(current_pixel)

			if current_pixel > local_max and paint_pos_check(j, painted_pos, abstracted_coord):
				local_max = current_pixel
				pos_cursor = j

		global_max += local_max
		painted_pos[pos_cursor] = 1
	"""
	return np.asarray(painted_pos)

def replica_rolling_ball(pixels):
	painted_pos = np.full(pixels.shape, 0).flatten().tolist()
	#abstracted_pixels_flat = abstracted_pixels.flatten().tolist()
	dim = pixels.shape

	print(dim)

	return np.asarray(painted_pos)

def deg_map_visualization(degs):
	xyz = []
	for i in range(0, degs.shape[0] - 1):
		for j in range(i + 1, degs.shape[0]):
			xyz.append([i, j, degs[i, j]])

	xyz = np.array(xyz)

	x=xyz[:,0]
	y=xyz[:,1]
	z=xyz[:,2]


	print(xyz.shape)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x,y,z)
	plt.show()

def deg_diag_visualization(degs):
	xyz = []
	for i in range(0, degs.shape[0] - 1):
		xyz.append(degs[i, i+1])

	plt.plot(xyz)
	plt.show()

def deg_svd(degs):
	xyz = []
	for i in range(0, degs.shape[0] - 1):
		for j in range(i + 1, degs.shape[0]):
			xyz.append([i, j, degs[i, j]])

	xyz = np.array(xyz)

	# SVD
	U, s, VT = svd(xyz)
	print(U)
	print(s)
	print(VT)

	# create m x n Sigma matrix
	Sigma = zeros((xyz.shape[0], xyz.shape[1]))
	# populate Sigma with n x n diagonal matrix
	Sigma[:xyz.shape[1], :xyz.shape[1]] = diag(s)
	# reconstruct matrix
	B = U.dot(Sigma.dot(VT))

	plt.imshow(B)
	plt.show()

def deg_map_normalizor(degs):
	d = []
	for i in range(0, degs.shape[0] - 1):
		for j in range(i + 1, degs.shape[0]):
			d.append([i, j, degs[i, j]])

	d = np.array(d)
	top = np.max(d[:, 2])
	bot = np.min(d[:, 2])

	if bot > 0:
		bot = 0

	x_normalizor = degs.shape[0]
	y_normalizor = degs.shape[1]
	z_normalizor = abs(bot) + top


	d[:, 2] = d[:, 2] + abs(bot)
	d[:, 0] = d[:, 0] / x_normalizor
	d[:, 1] = d[:, 1] / y_normalizor
	d[:, 2] = d[:, 2] / z_normalizor

	return d

def visualization(d):
	x=d[:,0]
	y=d[:,1]
	z=d[:,2]

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x,y,z)
	plt.show()

def grid_indexor(grid_size, val):
	idx = -1

	if val == 0:
		idx = 0
	else:
		interval = 1 / grid_size
		mod = val // interval

		if val % interval == 0:
			idx = mod - 1
		else:
			idx = mod

	if idx > grid_size - 1:
		idx = grid_size - 1

	return int(idx)

def grider(grid_size, d):
	flat = []
	grid_confid = np.zeros((grid_size, grid_size, grid_size))
	grid_shared = np.zeros((grid_size, grid_size, grid_size))
	k = d.shape[0]


	for i in range(k):
		x = grid_indexor(grid_size, d[i, 0])
		y = grid_indexor(grid_size, d[i, 1])
		z = grid_indexor(grid_size, d[i, 2])

		print(x)
		print(y)
		print(z)
		grid_confid[x, y, z] = grid_confid[x, y, z] + 1
		grid_shared[x, y, z] = 1 


	total = grid_confid.sum()

	for x in range(grid_size):
		for y in range(grid_size):
			for z in range(grid_size):
				flat.append(grid_confid[x, y, z] / total)
				grid_confid[x, y, z] = grid_confid[x, y, z] / total


	"""
	for i in range(grid_size):
		plt.imshow(grid_confid[:,:,i])
		#plt.imshow(grid_shared[i,:,:])
		plt.show()
	"""

	return grid_confid, grid_shared, flat






	#print(grid.shape)
	#print(grid)

def deg_horizontal_dist(degs):
	deg = []
	for i in range(0, degs.shape[0] - 1):
		for j in range(i + 1, degs.shape[0]):
			deg.append(degs[i, j])

	plt.plot(deg)
	plt.show()

def linear_rep(coord):
	K = np.zeros((len(coord), len(coord)))
	B = np.zeros((len(coord), len(coord)))
	KB = []

	for i in range(0, len(coord) - 1):
		for j in range(i + 1, len(coord)):
			a = coord[i]
			b = coord[j]
			X = np.array([a[0], b[0]]).reshape((-1, 1))
			y = np.array([a[1], b[1]])

			model = LinearRegression()
			model.fit(X, y)

			k = model.coef_.tolist()[0]
			b = model.intercept_

			K[i, j] = k
			B[i, j] = b
			KB.append([k, b])
	
	KB = np.array(KB)
	
	return K, B, KB

def vectorization(coord):
	vec = []

	for i in range(0, len(coord) - 1):
		for j in range(i + 1, len(coord)):
			a = coord[i]
			b = coord[j]

			dist = math.sqrt(((a[1] - b[1]) * (a[1] - b[1]) + (a[0] - b[0]) * (a[0] - b[0])))
			deg = math.degrees(math.atan((a[1] - b[1]) / (a[0] - b[0] + 0.001)))
			vec.append([a, b, dist, deg])
	
	vec = np.array(vec)
	
	return vec

def wise_vectorization(coord):
	vec = []

	for i in coord:
		temp = []
		for j in coord:
			if str(i) != str(j):
				dist = math.sqrt(((i[1] - j[1]) * (i[1] - j[1]) + (i[0] - j[0]) * (i[0] - j[0])))
				deg = math.degrees(math.atan((i[1] - j[1]) / (i[0] - j[0] + 0.001)))
				temp.append([i, j, dist, deg])
		vec.append(temp)
	vec = np.array(vec)
	
	return vec

def orig_vectorization(coord):
	vec = []

	for i in coord:
		dist = math.sqrt(((i[1]) * (i[1]) + (i[0]) * (i[0])))
		deg = math.degrees(math.atan((i[1]) / (i[0])))
		vec.append([i, dist, deg])

	vec = np.array(vec)
	
	return vec

def center_transformation(coord):
	coord = np.array(coord)
	c = [coord[:,0].mean(), coord[:,1].mean()]

	vec = []

	for i in coord:
		dist = math.sqrt(((i[1] - c[1]) * (i[1] - c[1]) + (i[0] - c[0]) * (i[0] - c[0])))
		deg = math.degrees(math.atan((i[1] - c[1]) / (i[0] - c[0] + 0.001)))
		vec.append([i, dist, deg])

	vec = np.array(vec)

	
	return vec

def init_abs(size):
	res = []
	coord = []
	res_tracker = []
	coord_tracker = []
	

	for i in range(size):
		for j in range(size):
			coord.append([i, j])
			coord_tracker.append([i, j])

	coord = np.array(coord).astype(float)

	x_min = coord[:, 0].min()
	x_max = coord[:, 0].max()
	y_min = coord[:, 1].min()
	y_max = coord[:, 1].max()
	x_scale = 1.0 / (x_max - x_min)
	y_scale = 1.0 / (y_max - y_min)

	coord[:, 0] = coord[:, 0] * x_scale
	coord[:, 1] = coord[:, 1] * y_scale

	"""
	coord = coord.tolist()

	for i in range(0, len(coord) - 1):
		for j in range(i+1, len(coord)):
	"""

	for i in range(0, coord.shape[0] - 1):
		for j in range(i+1, coord.shape[0]):
			x_c = (coord[i][0] - coord[j][0]) / 2 if coord[i][0] > coord[j][0] else (coord[j][0] - coord[i][0]) / 2
			y_c = (coord[i][1] - coord[j][1]) / 2 if coord[i][1] > coord[j][1] else (coord[j][1] - coord[i][1]) / 2
			m = math.sqrt(((coord[i][1] - coord[j][1]) * (coord[i][1] - coord[j][1]) + (coord[i][0] - coord[j][0]) * (coord[i][0] - coord[j][0])))
			m_norm = m / size
			d = math.degrees(math.atan((coord[i][1] - coord[j][1]) / (coord[i][0] - coord[j][0] + 0.001)))

			res.append([x_c, y_c, m_norm, d])
			res_tracker.append([coord_tracker[i], coord_tracker[j]])

	return np.array(res), np.array(res_tracker)

def preprocess(pixels):
	x = pixels.shape[0]
	y = pixels.shape[1]

	coord = []
	

	for i in range(x):
		for j in range(y):
			if pixels[i, j] == 1:
				coord.append([i, j])

	res = []

	coord = np.array(coord).astype(float)

	x_min = coord[:, 0].min()
	x_max = coord[:, 0].max()
	y_min = coord[:, 1].min()
	y_max = coord[:, 1].max()
	x_scale = 1.0 / (x_max - x_min)
	y_scale = 1.0 / (y_max - y_min)

	# remove the extra white spaces
	coord[:, 0] = coord[:, 0] - x_min
	coord[:, 1] = coord[:, 1] - y_min


	# normalize the coord
	coord[:, 0] = coord[:, 0] * x_scale
	coord[:, 1] = coord[:, 1] * y_scale

	"""
	for i in range(coord.shape[0]):
		for j in range(coord.shape[0]):
	"""
	for i in range(0, coord.shape[0] - 1):
		for j in range(i+1, coord.shape[0]):
			x_c = (coord[i][0] - coord[j][0]) / 2 if coord[i][0] > coord[j][0] else (coord[j][0] - coord[i][0]) / 2
			y_c = (coord[i][1] - coord[j][1]) / 2 if coord[i][1] > coord[j][1] else (coord[j][1] - coord[i][1]) / 2
			m = math.sqrt(((coord[i][1] - coord[j][1]) * (coord[i][1] - coord[j][1]) + (coord[i][0] - coord[j][0]) * (coord[i][0] - coord[j][0])))
			d = math.degrees(math.atan((coord[i][1] - coord[j][1]) / (coord[i][0] - coord[j][0] + 0.001)))
			

			res.append([x_c, y_c, m, d])


	return np.array(res)

def abs_builder(abst, img):
	conf = [0] * abst.shape[0]
	print(len(conf))


	for i in range(img.shape[0]):
		l2_min = 9999
		idx_min = -1

		for j in range(abst.shape[0]):
			x_c_delta = img[i][0] - abst[j][0]
			y_c_delta = img[i][1] - abst[j][1]
			m_delta = img[i][2] - abst[j][2]
			d_abs_i = abs(img[i][3] - abst[j][3])
			d_abs_ii = abs(img[i][3] - abst[j][3])
			d_delta = d_abs_i if d_abs_i < d_abs_ii else d_abs_ii
			d_delta = d_delta / 90 # normalize d_delta

			l2 = math.sqrt(x_c_delta * x_c_delta + y_c_delta * y_c_delta + m_delta * m_delta + d_delta * d_delta)
			
			if l2 < l2_min:
				l2_min = l2
				idx_min = j

		conf[idx_min] = conf[idx_min] + 1

	return conf

# using binary or avg to fill the new layers?
def above(depth, step_size, img):
	abs_img = img

	for i in range(depth):
		d = abs_img.shape[0]
		n_d = d - step_size
		temp = np.zeros(shape=(n_d, n_d))

		for m in range(d):
			for n in range(d):
				n_r = int((m) * (1/d) / (1/n_d))
				n_c = int((n) * (1/d) / (1/n_d))
				temp[n_r, n_c] = abs_img[m, n]


		abs_img = temp

		fig, axs = plt.subplots(2)
		fig.suptitle("Image - Plot")
		axs[0].imshow(img)
		axs[1].imshow(abs_img)
		plt.show()

def feature_filter(pixels):
	x = pixels.shape[0]
	y = pixels.shape[1]

	filter_size = 3 # it should be changed as feature filter size changed

	feature = {
		"u0": np.array([[-1, -1, -1], [1, 1, 1], [1, 1, 1]]), # -
		#"l0": np.array([[1, 1, 1], [1, 1, 1], [-1, -1, -1]]), # -
		"u135": np.array([[1, -1, -1], [1, 1, -1], [1, 1, 1]]), # \
		#"l135": np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1]]) # \
		"u90": np.array([[-1, 1, 1], [-1, 1, 1], [-1, 1, 1]]), # |
		#"l90": np.array([[1, 1, -1], [1, 1, -1], [1, 1, -1]]), # |
		"u45": np.array([[-1, -1, 1], [-1, 1, 1], [1, 1, 1]]), # /
		#"l45": np.array([[1, 1, 1], [1, 1, -1], [1, -1, -1]]), # /
	}



	feature = {
		#"u0": np.array([[-1, -1, -1], [1, 1, 1], [1, 1, 1]]), # -
		"l0": np.array([[1, 1, 1], [1, 1, 1], [-1, -1, -1]]), # -
		#"u135": np.array([[1, -1, -1], [1, 1, -1], [1, 1, 1]]), # \
		"l135": np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1]]), # \
		#"u90": np.array([[-1, 1, 1], [-1, 1, 1], [-1, 1, 1]]), # |
		"l90": np.array([[1, 1, -1], [1, 1, -1], [1, 1, -1]]), # |
		#"u45": np.array([[-1, -1, 1], [-1, 1, 1], [1, 1, 1]]), # /
		"l45": np.array([[1, 1, 1], [1, 1, -1], [1, -1, -1]]), # /
	}

	feature_map = []

	for k in feature.keys():
		f = []

		for i in range(x - filter_size):
			for j in range(y - filter_size):
				temp = np.multiply(pixels[i: i + filter_size, j: j + filter_size], feature[k]).sum() / (filter_size * filter_size)
				f.append(temp)
		
		l = len(f)
		s = int(math.sqrt(l))
		f = np.array(f, dtype='float').reshape((s, s))

		feature_map.append(f)

	return feature_map

def feature_maxpool(features, size):
	maxpool_map = []
	boundary = features[0].shape[0] - features[0].shape[0] % size

	for k in features:
		m = []

		for i in np.arange(0, boundary, size):
			for j in np.arange(0, boundary, size):
				temp = k[i: i + size, j: j + size].max()
				if temp >= 0.9:
					m.append(temp)
				else:
					m.append(0)

		l = len(m)
		s = int(math.sqrt(l))
		m = np.array(m, dtype='float').reshape((s, s))

		maxpool_map.append(m)

	return maxpool_map

def definition_builder(features):
	# build a ele-wise relatively correlated rule matrix to reflect
	# the internal relationship which indicate the "feeling" of seeing img


	c = []
	for k in range(len(features)):
		feature = features[k]
		for i in range(feature.shape[0]):
			for j in range(feature.shape[1]):
				if feature[i, j] > 0:
					c.append([i, j, k])

	t = np.zeros((4, 4, 8))


	print(c)

	for i in c:
		for j in c:
			temp = relative_pos(i, j)

			if temp != -1:
				t[int(i[2]), int(j[2]), temp] = t[int(i[2]), int(j[2]), temp] + 1



	# 1. for each feature, get the coords
	# 2. for each feature, using coords to construct a ele-wise relative position map
	# 3. using this position map, discover some "repeatly" occuring general patterns,
	# such as "circle", "left-curve", "vertical straight line", etc.
	"""
	"""


	return t

"""
def relative_pos(current, relative):
	l = 0
	r = 4
	u = 2
	d = 6

	ul = 1
	ur = 3
	dl = 7
	dr = 5

	relative = [relative[0] - current[0], relative[1] - current[1]]

	#print(relative)

	rp = ""

	if relative[0] == 0 and relative[1] == 0:
		return -1

	if relative[0] == 0:
		if relative[1] > 0:
			rp = u
		else:
			rp = d

	if relative[1] == 0:
		if relative[0] > 0:
			rp = r
		else:
			rp = l

	if relative[0] > 0 and relative[1] > 0:
		rp = ur
	elif relative[0] < 0 and relative[1] > 0:
		rp = ul
	elif relative[0] < 0 and relative[1] < 0:
		rp = dl
	elif relative[0] > 0 and relative[1] < 0:
		rp = dr

	#print(rp)

	return rp
"""
def relative_pos(current, relative):
	l = 0
	ul = 1
	u = 2
	ur = 3
	r = 4
	dr = 5	
	d = 6
	dl = 7


	relative = [relative[0] - current[0], relative[1] - current[1]]

	#print(relative)

	rp = ""

	if relative[0] == 0 and relative[1] == 0:
		return -1

	if relative[0] == 0: # x
		if relative[1] > 0: # y
			rp = r
		else:
			rp = l

	if relative[1] == 0: # y
		if relative[0] > 0: # x
			rp = d
		else:
			rp = u

	if relative[0] > 0 and relative[1] > 0:
		rp = dr
	elif relative[0] < 0 and relative[1] > 0:
		rp = ur
	elif relative[0] < 0 and relative[1] < 0:
		rp = ul
	elif relative[0] > 0 and relative[1] < 0:
		rp = dl

	#print(rp)

	return rp

"""
def prep_training_data(data_idxes):
	mndata = MNIST('dataset')
	images_train, labels_train = mndata.load_training()
	images_test, labels_test = mndata.load_testing()

	data = []

	for i in data_idxes:
		img = images_train[i]
		img = np.array(img, dtype='float').reshape((28, 28))
		img_bin = binarization(img)
		canvas = np.zeros(shape=(28, 28))
		
		for i in range(len(img_bin)):
			canvas[img_bin[i][0]][img_bin[i][1]] = 1

		data.append(canvas)

	return data

def prep_testing_data_labeled(data_idxes):
	mndata = MNIST('dataset')
	images_train, labels_train = mndata.load_training()
	images_test, labels_test = mndata.load_testing()

	data = []

	for i in data_idxes:
		img = images_test[i]
		img = np.array(img, dtype='float').reshape((28, 28))
		img_bin = binarization(img)
		canvas = np.zeros(shape=(28, 28))
		
		for i in range(len(img_bin)):
			canvas[img_bin[i][0]][img_bin[i][1]] = 1

		data.append(canvas)

	return data

def prep_testing_data(imgs):
	prep_imgs = []

	for i in imgs:
		img = np.array(img, dtype='float').reshape((28, 28))
		img_bin = binarization(img)
		canvas = np.zeros(shape=(28, 28))
		
		for i in range(len(img_bin)):
			canvas[img_bin[i][0]][img_bin[i][1]] = 1

		prep_imgs.append(canvas)

	return prep_imgs

def get_testing_data_labeled(label, num_pick):
	mndata = MNIST('dataset')
	images_train, labels_train = mndata.load_training()
	images_test, labels_test = mndata.load_testing()


	labels_test = np.array(labels_test)
	idxes = np.where(labels_test == label)[0].tolist()
	size = len(idxes) - 1

	idx = []

	for i in range(num_pick):
		idx.append(idxes[random.randint(0, size)])

	return idx

def check_data(label, num_pick):
	mndata = MNIST('dataset')
	images_train, labels_train = mndata.load_training()
	images_test, labels_test = mndata.load_testing()


	labels_train = np.array(labels_train)
	idxes = np.where(labels_train == label)[0].tolist()
	size = len(idxes) - 1

	idx = []

	for i in range(num_pick):
		idx.append(idxes[random.randint(0, size)])

	return idx
"""















def prep_training_data(data_idxes):
	mndata = MNIST('dataset')
	images_train, labels_train = mndata.load_training()
	images_test, labels_test = mndata.load_testing()

	data = []

	for i in data_idxes:
		img = images_train[i]
		img = np.array(img, dtype='float').reshape((28, 28))
		img_bin = binarization(img)
		canvas = np.zeros(shape=(28, 28))
		
		for i in range(len(img_bin)):
			canvas[img_bin[i][0]][img_bin[i][1]] = 1

		data.append(canvas)

	return data

def prep_testing_data_labeled(data_idxes):
	mndata = MNIST('dataset')
	images_train, labels_train = mndata.load_training()
	images_test, labels_test = mndata.load_testing()

	data = []

	for i in data_idxes:
		img = images_test[i]
		img = np.array(img, dtype='float').reshape((28, 28))
		img_bin = binarization(img)
		canvas = np.zeros(shape=(28, 28))
		
		for i in range(len(img_bin)):
			canvas[img_bin[i][0]][img_bin[i][1]] = 1

		data.append(canvas)

	return data

def prep_testing_data(imgs):
	prep_imgs = []

	for i in imgs:
		img = np.array(img, dtype='float').reshape((28, 28))
		img_bin = binarization(img)
		canvas = np.zeros(shape=(28, 28))
		
		for i in range(len(img_bin)):
			canvas[img_bin[i][0]][img_bin[i][1]] = 1

		prep_imgs.append(canvas)

	return prep_imgs

def get_testing_data_labeled(label, num_pick):
	mndata = MNIST('dataset')
	images_train, labels_train = mndata.load_training()
	images_test, labels_test = mndata.load_testing()


	labels_test = np.array(labels_test)
	idxes = np.where(labels_test == label)[0].tolist()
	size = len(idxes) - 1

	idx = []

	for i in range(num_pick):
		idx.append(idxes[random.randint(0, size)])

	return idx

def check_data(label, num_pick):
	mndata = MNIST('dataset')
	images_train, labels_train = mndata.load_training()
	images_test, labels_test = mndata.load_testing()


	labels_train = np.array(labels_train)
	idxes = np.where(labels_train == label)[0].tolist()
	size = len(idxes) - 1

	idx = []

	for i in range(num_pick):
		idx.append(idxes[random.randint(0, size)])

	return idx















"""
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
>>> # y = 1 * x_0 + 2 * x_1 + 3
>>> y = np.dot(X, np.array([1, 2])) + 3
>>> reg = LinearRegression().fit(X, y)
>>> reg.score(X, y)
1.0
>>> reg.coef_
array([1., 2.])
>>> reg.intercept_
3.0000...
>>> reg.predict(np.array([[3, 5]]))
"""











