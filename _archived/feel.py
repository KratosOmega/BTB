
from random import *
from math import *
import time
import numpy as np
import copy

"""
1. 	feeling clock is rotating, all features has specific time of staying,
	corresponding feature will be giving more time of staying. While staying,
	a random.uniform(0, 1) will be adding values to the corresponding feature 
	pool. At the end, feature with highest value in corresponding feature pool 
	is the "right feeling".
2. 	the basic feeling in this proj. is 4 (idx: 0, 1, 2, 3) in total, but the 
	feeling clock should be sliced with more details, such as 12 or 24 in total.
"""

class Feel(object):
	def __init__(self, learning_rate, forgeting_rate, feel_num, feel_detail_num, feel_time, time_size):
		self.lr = learning_rate # define the weight of learning
		self.fr = forgeting_rate # define the weight of forgeting
		self.fn = feel_num # number of different feelings
		self.fdn = feel_detail_num
		self.default_time = 1
		#self.e = epsilon # random scale for timer, range in [0, 1]

		self.hourglass = [0] * feel_num * feel_detail_num # feel weight
		self.fc = [1] * feel_num * feel_detail_num # feel wheel / clock
		self.td = self.time_distribution() # time distribution
		self.total_sand = np.array(self.td)[:, 0].sum()
		self.fs = [1] * feel_num # feel speed
		self.ts = time_size

		self.feel = self.init_feel()
		


		self.t = feel_time * feel_num
		#self.t_w_p = self.time_passed_w_penalty()
		#self.t_r = uniform(0, feel_num)

	def init_feel(self):
		return False


	# TODO: the randomness of time passed may not be very well designed,
	# need to come back to double check!!!
	"""
	def time_passed(self):
		start = time.time()
		r = -1

		while r < self.e:
			r = uniform(0, 1)

		end = time.time()

		return end - start

	def time_passed_w_penalty(self):
		start = time.time()

		t_penalty = 0
		r = -1

		while r < self.e:
			r = uniform(0, 1)
			t_penalty = t_penalty + uniform(0, 1)

		end = time.time()

		return (end - start) + t_penalty
	"""

	def time_distribution(self):
		"""
		[
			[time_alloc, next_idx, pre_idx],
			[time_alloc, next_idx, pre_idx],
			...
		]
		"""
		td = []

		for i in range(self.fn * self.fdn):
			td.append([self.default_time, i + 1, i - 1])

		td[-1][1] = td[-1][1] - len(td) # connect the last ele to the first ele
		td[0][2] = td[0][2] + len(td) # connect the frist ele to the last ele

		return td




	def feelit(self, feature):
	
		return False





	"""
	TODO:
	1. 	add depreciation (forgetting) over a certain period of time, 
		if certain feature is not encountered
	"""
	def update(self, feature):
		spread_size = 1

		# fn = [...,   3,    0,    1,    2,    3,    0, ...]
		# td = [...,5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, ...]
		central_idx = feature * self.fdn
		idx_to_update = [central_idx]
		idxs = np.array(self.td)[:, 1:3]

		# to left
		cur_idx = central_idx
		for i in range(spread_size):
			idx_to_update.append(idxs[cur_idx, 0])
			cur_idx = idxs[cur_idx, 0]

		# to right
		cur_idx = central_idx
		for i in range(spread_size):
			idx_to_update.append(idxs[cur_idx, 1])
			cur_idx = idxs[cur_idx, 1]

		total_sand = np.array(self.td)[:, 0].sum()
		add_sand = total_sand * self.lr / len(idx_to_update)
		ded_sand = total_sand * self.lr / (len(self.td) - len(idx_to_update))


		for i in range(len(self.td)):
			if i in idx_to_update:
				self.td[i][0] = self.td[i][0] + add_sand

			else:
				self.td[i][0] = self.td[i][0] - ded_sand
				if self.td[i][0] < 0:
					self.td[i][0] = 0

		self.td = np.array(self.td)
		normalizor = self.td[:, 0].sum() / self.total_sand

		self.td[:, 0] = self.td[:, 0] / normalizor

		#print(self.td[:, 0].sum())

		self.td = self.td.tolist()


	"""
	TODO: 
	1.	impression() has a problem of favoring the left-most section, 
		need to fix this (make it relatively uniformed)
	"""
	def impression(self):
		idx = 0
		t = self.t
		td = copy.deepcopy(self.td)
		hourglass = [0] * self.fn * self.fdn

		while t > 0:
			if td[idx][0] >= self.ts:
				sand = uniform(0, 1)
				hourglass[idx] = hourglass[idx] + sand
				td[idx][0] = td[idx][0] - self.ts
				t = t - self.ts
			else:
				td[idx][0] = self.td[idx][0]
				idx = int(td[idx][1])

		hourglass = np.array(hourglass)
		total_sand = hourglass.sum()
		hourglass = hourglass / total_sand

		return hourglass

	

	

























