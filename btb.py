from random import *
from math import *
import time
import numpy as np
import copy
from matplotlib import pyplot as plt
"""
1. 	feeling clock is rotating, all features has specific time of staying,
	corresponding feature will be giving more time of staying. While staying,
	a random.uniform(0, 1) will be adding values to the corresponding feature 
	pool. At the end, feature with highest value in corresponding feature pool 
	is the "right feeling".
2. 	the basic feeling in this proj. is 4 (idx: 0, 1, 2, 3) in total, but the 
	feeling clock should be sliced with more details, such as 12 or 24 in total.



3. 2 ways of doing abstraction:
	1). mention all the differences, but ignore the similarities.
		etc: if d1, d2, d3, d4, ...., then it's Class A
	2). mention all the similarities, but igonore the differentces.
		etc: if s1, s2, s3, s4, ...., then it's Class B


4.  traditional machine learning: using few data to build the proto-type model, which 
	covers the "similarities", as well as "differences". Then the more data is fed, 
	the more "differences" is eliminated from the model. At the end, the furnished model
	is reaching a status of having majority of similarities, with very few "differences".

	what i suggested, is IF we know the solution, then we define what the solution should be.



5.  for img data:
	human never using the way of computer to process the img.

	for e.g., in numerical data, human can not identify the pattern, so human has to use
	math to plot the pattern into plot, which is "img" to identify pattern.

	but, when we only focus on img data, human already jump to the img part, which makes human
	very easy to see the pattern. However, computer has to start from the raw data stage to process
	all the way to the "img stage".

	what make human can jump from raw data directly to img data directly is the key!



!!!: 
	a) define basic logic blocks.
	b) re-represent the features with basic logic blocks.
	c) re-constructed features should share the same logics for the same img.


6. create feature relative location map (gorup with adjacent features)



"""


# Bottom to Top, Top to Bottom

class BTB(object):
	def __init__(self, eye_size, orig_img):
		self.eye_size = eye_size
		self.orig_img = orig_img.astype(int)
		self.lvl0 = False
		self.lvl1 = False
		self.lvl2 = False
		self.lvl3 = False
		self.lvl4 = False
		self.patterns = self.pattern_generator(self.eye_size)

	# TODO: currently fixed size = 2, enable size-variation in future
	def pattern_generator(self, size):
		p = [
			"0000",
			"0001",
			"0010",
			"0011",
			"0100",
			"0101",
			"0110",
			"0111",
			"1000",
			"1001",
			"1010",
			"1011",
			"1100",
			"1101",
			"1110",
			"1111",
		]

		return p

	def binary_max(self, size, b_size):
		l = size * size * b_size
		b = ""
		for i in range(l):
			b = b + "1"
		boundary = float(int(b, 2))

		return boundary

	def rp(self, img, pattern):
		res = []
		x = img.shape[0]
		y = img.shape[1]

		for i in range(x):
			for j in range(y):
				if img[i, j] != pattern:
					res.append(0)
				else:
					res.append(1)

		return np.array(res, dtype='int').reshape(img.shape)

	def maxpool(self, img, size):
		x = img.shape[0]
		y = img.shape[1]
		m = []

		for i in np.arange(0, x, size):
			for j in np.arange(0, y, size):
				#print(str(i) + "-" + str(j))

				temp = img[i: i + size, j: j + size].max()
				m.append(temp)

		l = len(m)
		s = int(sqrt(l))
		m = np.array(m).reshape((s, s))

		return m

	def abstracting(self, img):
		size = self.eye_size
		shape = img.shape
		b_size = len(str(img[0, 0]))
		b_max = self.binary_max(size, b_size)

		temp_b = []
		temp_d = []

		for i in range(shape[0] - size + 1):
			for j in range(shape[1] - size + 1):
				focus = img[i: i + size, j: j + size]
				focus_flat = [str(i) for i in focus.flatten()]
				b = "".join(focus_flat)
				temp_b.append(b)
				d = float(int(b,2))
				temp_d.append(d / b_max)

		res_b = np.array(temp_b, dtype='str').reshape((shape[0] - size + 1, shape[1] - size + 1))
		res_d = np.array(temp_d, dtype='float').reshape((shape[0] - size + 1, shape[1] - size + 1))

		return res_b, res_d

	def pattern_analysis(self, data, prefix = ""):
		pattern_map = {}
		pattern_ratio = []
		data_size = data.shape[0] * data.shape[1]

		# extract all basic patterns with cnn filters (16 basic patterns)
		binary_map, d = self.abstracting(data)

		for p in self.patterns:
			# get map of a single pattern
			pm = self.rp(binary_map, p)

			# pattern ratio
			#pr = float(pm.sum()) / float(data_size)
			pr = 1 if pm.sum() > 0 else 0
			pattern_ratio.append(pr)

			# maxpool to reduce pattern map size (further abstracting)
			pm_mp = self.maxpool(pm, 2)

			# seperate basic patterns from combo (1) into individual maps (16)
			pattern_map[prefix + p] = pm_mp

		return pattern_map, pattern_ratio

	def perception(self):
		print("...")
		# ================================================
		#                     LVL - 0
		# ================================================
		data = self.orig_img
		pm_0, pr_0 = self.pattern_analysis(data)


		"""
		fig, axs = plt.subplots(5, 4)
		axs[0, 0].imshow(pm_0["0000"])
		axs[0, 1].imshow(pm_0["0001"])
		axs[0, 2].imshow(pm_0["0010"])
		axs[0, 3].imshow(pm_0["0011"])
		axs[1, 0].imshow(pm_0["0100"])
		axs[1, 1].imshow(pm_0["0101"])
		axs[1, 2].imshow(pm_0["0110"])
		axs[1, 3].imshow(pm_0["0111"])
		axs[2, 0].imshow(pm_0["1000"])
		axs[2, 1].imshow(pm_0["1001"])
		axs[2, 2].imshow(pm_0["1010"])
		axs[2, 3].imshow(pm_0["1011"])
		axs[3, 0].imshow(pm_0["1100"])
		axs[3, 1].imshow(pm_0["1101"])
		axs[3, 2].imshow(pm_0["1110"])
		axs[3, 3].imshow(pm_0["1111"])
		axs[4, 0].imshow(data)
		plt.show()
		"""



		# lvl1 abstraction, shape (1, 16)
		lvl0 = np.array(pr_0)

		self.lvl0 = lvl0
		
		"""
		lvl0[0] = 0
		lvl0[-1] = 0
		plt.plot(lvl0)
		plt.show()
		"""


		print("--------------------------")

		# ================================================
		#                     LVL - 1
		# ================================================
		lvl1 = np.zeros((16, 16), dtype = np.float)
		pm_1 = {}

		for i in self.patterns:
			data = pm_0[i]
			temp_pm_1, pr_1 = self.pattern_analysis(data, i)

			pm_1.update(temp_pm_1)
			pr_1 = np.array(pr_1)

			idx_0 = int(i,2)
			lvl1[idx_0, :] = pr_1
		
			"""
			fig, axs = plt.subplots(5, 4)
			fig.suptitle(i)
			axs[0, 0].imshow(pm_1["0000"])
			axs[0, 1].imshow(pm_1["0001"])
			axs[0, 2].imshow(pm_1["0010"])
			axs[0, 3].imshow(pm_1["0011"])
			axs[1, 0].imshow(pm_1["0100"])
			axs[1, 1].imshow(pm_1["0101"])
			axs[1, 2].imshow(pm_1["0110"])
			axs[1, 3].imshow(pm_1["0111"])
			axs[2, 0].imshow(pm_1["1000"])
			axs[2, 1].imshow(pm_1["1001"])
			axs[2, 2].imshow(pm_1["1010"])
			axs[2, 3].imshow(pm_1["1011"])
			axs[3, 0].imshow(pm_1["1100"])
			axs[3, 1].imshow(pm_1["1101"])
			axs[3, 2].imshow(pm_1["1110"])
			axs[3, 3].imshow(pm_1["1111"])
			axs[4, 0].imshow(data)
			plt.show()
			"""

		#print(lvl1)

		#plt.imshow(lvl1)
		#plt.show()

		# ignore [0, 15] patterns (none and all)
		for a in range(16):
			for b in range(16):
				if a in [0, 15] or b in [0, 15]:
					lvl1[a, b] = 0

		self.lvl1 = lvl1




		
		# ================================================
		#                     LVL - 2
		# ================================================
		lvl2 = np.zeros((16, 16, 16), dtype = np.float)
		pm_2 = {}

		for i in self.patterns:
			for j in self.patterns:
				data = pm_1[i+j]

				temp_pm_2, pr_2 = self.pattern_analysis(data, i+j)

				pm_2.update(temp_pm_2)
				pr_2 = np.array(pr_2)
				
				idx_0 = int(i,2)
				idx_1 = int(j,2)
				lvl2[idx_0, idx_1, :] = pr_2

		# ignore [0, 15] patterns (none and all)
		for a in range(16):
			for b in range(16):
				for c in range(16):
					if a in [0, 15] or b in [0, 15] or c in [0, 15]:
						lvl2[a, b, c] = 0

		self.lvl2 = lvl2


		# ================================================
		#                     LVL - 3
		# ================================================
		lvl3 = np.zeros((16, 16, 16, 16), dtype = np.float)
		pm_3 = {}

		for i in self.patterns:
			for j in self.patterns:
				for k in self.patterns:
					data = pm_2[i+j+k]

					temp_pm_3, pr_3 = self.pattern_analysis(data, i+j+k)

					pm_3.update(temp_pm_3)
					pr_3 = np.array(pr_3)
					
					idx_0 = int(i,2)
					idx_1 = int(j,2)
					idx_2 = int(k,2)
					lvl3[idx_0, idx_1, idx_2, :] = pr_3

		# ignore [0, 15] patterns (none and all)
		for a in range(16):
			for b in range(16):
				for c in range(16):
					for d in range(16):
						if a in [0, 15] or b in [0, 15] or c in [0, 15] or d in [0, 15]:
							lvl3[a, b, c, d] = 0

		self.lvl3 = lvl3
		"""
		"""















		"""
		# ================================================
		#                     LVL - 4
		# ================================================
		lvl4 = np.zeros((16, 16, 16, 16, 16), dtype = np.float)
		pm_4 = {}

		for i in self.patterns:
			for j in self.patterns:
				for k in self.patterns:
					for h in self.patterns:
						data = pm_3[i+j+k+h]

						temp_pm_4, pr_4 = self.pattern_analysis(data, i+j+k+h)

						pm_4.update(temp_pm_4)
						pr_4 = np.array(pr_4)
						
						idx_0 = int(i,2)
						idx_1 = int(j,2)
						idx_2 = int(k,2)
						idx_3 = int(h,2)
						lvl4[idx_0, idx_1, idx_2, idx_3, :] = pr_4

		# ignore [0, 15] patterns (none and all)
		for a in range(16):
			for b in range(16):
				for c in range(16):
					for d in range(16):
						for e in range(16):
							if a in [0, 15] or b in [0, 15] or c in [0, 15] or d in [0, 15] or e in [0, 15]:
								lvl3[a, b, c, d, e] = 0


			self.lvl4 = lvl4
		"""

































