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


# NOT ONLY, BUT ONLY
# Different in attributes, but Same in ideas
class NOBO(object):
	def __init__(self):
		self.block_patterns = np.array([[2, 3, 4], [1, -1, 1], [4, 3, 2]])
		

	def init_abstraction(self):
		return False

	def comparison(self, A, B):
		res = 0
		C = A - B

		c = C.flatten().tolist()
		total = len(c)

		for i in c:
			if i == 0:
				res = res + 1

		res = float(res) / float(total)
		return res

	def get_block_patterns(self):
		res = []
		pattern = np.array([
			[0, 0, 0],
			[0, 0, 0],
			[0, 0, 0],
		])

		for i in range(pattern.shape[0]):
			for j in range(pattern.shape[1]):
				coord = [i, j]
				m = coord[0] - 1 + 0.0001
				n = coord[1] - 1 + 0.0001
				temp = round(m / n)
	
				if coord[0] == 1 and coord[1] == 1:
					res.append(-1)
				elif temp == 1:
					res.append(45)
				elif temp == 0:
					res.append(0)
				elif temp == -1:
					res.append(135)
				else:
					res.append(90)

		return res
		


	# this logic consider +/- of the degrees, which means
	# 0,0 + 1,1 = 1,1 + 2,2
	# may need to split these into 2 groups in the future if it doesn't work well
	def merge_logic(self, pattern):
		"""
		pattern is a 3x3 sub-matrix, that illustrate the patterns layout
		"""
		"""
		[0. 0. 0. 0. 0. 0.]
 		[0. 0. 0. 0. 3. 0.]
		[0. 0. 1. 1. 3. 0.]
		[0. 0. 0. 3. 0. 0.]
		[0. 0. 0. 3. 0. 0.]
		[0. 0. 0. 0. 0. 0.]


		[0. 0. 0. 0. 0. 0.]
 		[0. 0. 0. 0. |. 0.]
		[0. 0. -. -. |. 0.]
		[0. 0. 0. |. 0. 0.]
		[0. 0. 0. |. 0. 0.]
		[0. 0. 0. 0. 0. 0.]


		[0. 0. 0. 0. 0. 0.]
 		[0. 0. 0. 0. 0. 0.]
 		[0. 0. -. -. 0. |.]
 		[0. -. 0. 0. /. 0.]
 		[0. 0. 0. /. /. 0.]
 		[0. 0. /. /. 0. 0.]






		[0, 0, 0, 0],
		[0, 0, 0, 0],
		[0, 0, 0, 0],
		[0, 0, 0, 0],

		[00, 01, 02, 03],
		[10, 11, 12, 13],
		[20, 21, 22, 23],
		[30, 31, 32, 33],

		







		[0, 0, 3],
		[1, 1, 3],
		[0, 3, 0],
		[0.0, 0.35294117647058826, 0.11764705882352941, 0.4117647058823529, 0.11764705882352941]


		[0, 0, 0],
		[0, 1, 1],
		[0, 0, 3],
		[0.0, 0.4, 0.13333333333333333, 0.3333333333333333, 0.13333333333333333]
		
		"""


		"""
		pattern = np.array([
			[0, 0, 0],
			[1, 1, 1],
			[0, 0, 0],

			#[0, 4, 4],
			#[4, 4, 0],
			#[4, 0, 0],
		])
		"""

		# 0 = none, 1 = 0, 2 = 45, 3 = 90, 4 = 135
		output = [0] * 5

		for i in range(pattern.shape[0]):
			for j in range(pattern.shape[1]):
				if self.block_patterns[i, j] > -1 and pattern[i, j] != 0:
					output[self.block_patterns[i, j]] = output[self.block_patterns[i, j]] + 1
				if self.block_patterns[i, j] % 2 == 0 and pattern[i, j] != 0:
					output[1] = output[1] + 0.5
					output[3] = output[3] + 0.5

				output[int(pattern[i, j])] = output[int(pattern[i, j])] + 1
		"""
		currently disable the consideration of empty spaces
		"""
		output[0] = 0
		#print(output)

		total = sum(output) + 0.00000001

		for i in range(len(output)):
			output[i] = output[i] / total
		# -------------------------------------------------
		#print(output)
		#print(output.index(max(output)))
		return output

	def get_33_matrix(self, A, center):
		return False





	def def_line(self, inp):
		"""
		concept: a list of analogies
		analogy: a similarity




		[0. 0. 0. 0. 0. 0.]
 		[0. 0. 0. 0. 3. 0.]
		[0. 0. 1. 1. 3. 0.]
		[0. 0. 0. 3. 0. 0.]
		[0. 0. 0. 3. 0. 0.]
		[0. 0. 0. 0. 0. 0.]



		from above matrix, find the all possibles of having 2 lines, which means
		split the above no-zeros into 2 groups

		"""

		# get coord & features
		temp = []

		for i in range(inp.shape[0]):
			for j in range(inp.shape[1]):
				if inp[i, j] != 0:
					temp.append([i, j, inp[i, j]])

		print(temp)

		# split into 2 groups (2 lines)


		print(inp)




	def logic_definition(self):
		"""
		what is 7?

		2 relative "straight lines" intersect and stop in and at a 0-90 degree angle

		1. 2 relative "straight lines"
		2. intersect and stop in and at
		3. a 0-90 degree angle


		create a rule that can do self-auto-update: add, remove, modify
		"""




		# define a basic logic of viewing multiple pattern into a single one (new or old)
		merge_logic = {
			1: np.array([[-1, 1, -1],[-1, 1, -1], [-1, 1, -1]]),
			45: np.array([[-1, -1, 1],[-1, 1, -1], [1, -1, -1]]),
			-1: np.array([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]),
		}
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






















