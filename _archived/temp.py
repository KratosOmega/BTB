"""
Author: XIN LI

TODO:
* Derive a "abstractive definition", which can be automatically generated by model.
* such "abstractive definition" (using [Probability Rules] to represent) should be:

abstractive:
- human: cover all humen, but can not be used to represent a specific human (black, white, yellow)
- color: cover all colors, but can not be used to represent a specific color (green, red, yellow)
- seven, cover all sevens, but can not be used to represent a specific hand-writing "7"

definition:
- human: leg, arm, heads; but all these components are allow to be different and yet similar in some ways




Using playing saxophone as an example.

A particular "KEY" is hard-defined to be a KEY on the saxophone. But every time you play the KEY, it will be 
different (similar, but never the same as last time, due to the air blow).

So, how can i setup a similar hard-defined KEY in my model (instead of on the sax)?







inheritable difference: 


Normalize the scale to the same:
- X => 0 ~ 1
- Y => 0 ~ 1
- X => min ~ max





what's the meaning of "alike"?
- in statistics, we use "distance" to evaluate the "likeness"
- in abstraction, we can try to use "substitutability" to evaluate the "likeness"






when you close your eyes, imagin a pattern, you see it, but actually you didn't,
because it's actually all black when you close your eyes. it is the brain that
read and give you a memory of such pattern to "make you think" you see it.

the amazing part is that, since there is not actual pattern to relfect the light into
eyes to see such pattern, and human can not remember the "pixel values" to remember the pattern,
how we store such pattern in our memory and read it out when we close our eyes?
"""
# ---------------------------------- customized libs
from utils import *
from feel import Feel
from nobo import NOBO
from btb import BTB
# ---------------------------------- open-source libs
from mnist import MNIST
from matplotlib import pyplot as plt
import numpy as np
import random
import math
from PIL import Image
import statistics


from mpl_toolkits.mplot3d import Axes3D

# load dataset
#idx = 6 # 1
#idx = 5 # 2
#idx = 16 # 2
#idx = 20 # 2
#idx = 12 # 3





#idx = 15 # 7




#idx = 29 # 7
#idx = 38 # 7
#idx = 42 # 7
#idx = 52 # 7
#idx = 71 # 7
#idx = 79 # 7
#idx = 84 # 7


#idx = 7 # 3
#idx = 91 # 7
idx = 96 # 7


mndata = MNIST('dataset')
images_train, labels_train = mndata.load_training()
images_test, labels_test = mndata.load_testing()


"""
# --------------------------------- Load an perfect example img
img = Image.open("./example.png")
img = img.resize((28,28),Image.ANTIALIAS)
img = np.array(img, dtype='float').reshape((28, 28))
#plt.imshow(img.reshape((28, 28)))
#plt.show()
"""

# --------------------------------- load hand-writing data
img = images_train[idx]
img = np.array(img, dtype='float').reshape((28, 28))

real_coord1 = binarization(img)

#img = images_train[6] # 1
#img = images_train[16] # 2
#img = images_train[12] # 3
img = images_train[91] # 7
#img = images_train[29] # 7
#img = images_train[42] # 7
#img = images_train[96] # 7

img = np.array(img, dtype='float').reshape((28, 28))

real_coord2 = binarization(img)


#coord1 = get_fake_coord()

"""
coord1 = np.array([
		[5, 7],
		[5, 14],
		[14, 7],
		[14, 14],
	])


coord2 = np.array([
		[5, 7],
		[14, 7],
		[5, 14],
		[14, 14],
	])
"""


"""
coord1 = np.array([
		[5, 7],
		[5, 14],
		[14, 7],
		[14, 14],


		[4, 9],
		[4, 11],
		[4, 13],
		[7, 13],
		[9, 11],
		[11, 9],
		[14, 9],
		[14, 11],
	])

coord2 = np.array([
		[14, 14],
		[14, 7],
		[5, 14],
		[11, 14],
		[8, 7],
		[11, 7],
		[8, 14],
		[14, 11],
		[14, 17],
		[18, 14],
		[14, 9],
		[5, 7],
		
	])
"""

"""
coord1 = np.array([
		[5, 7],

		[4, 9],
		[4, 11],
		[4, 13],

		[5, 14],

		[7, 13],
		[9, 11],
		[11, 9],

		[14, 7],

		[14, 9],
		[14, 11],

		[14, 14],
	])


coord2 = np.array([
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
	])
"""




"""
coord1 = np.array([
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

	])

coord2 = np.array([
		[2, 11],
		
		[5, 12],
		[11, 8],
		[6, 11],
		[5, 4],
		[4, 5],
		[3, 7],
		[2, 9],
		[8, 10],
		[9, 9],
		[3, 12],
		[13, 7],

	])
"""

"""

coord1 = np.array([
		[5, 7],

		[4, 9],
		[4, 11],
		[4, 13],

		[5, 14],

		[7, 13],
		[9, 11],
		[11, 9],

		[14, 7],

		[14, 9],
		[14, 11],

		[14, 14],
	])

coord2 = np.array([
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

	])
"""


coord1 = np.array([
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

	])

coord2 = np.array([
		[6, 5],
		[6, 7],
		[6, 9],
		[6, 11],
		[6, 13],
		[6, 15],
		[8, 15],
		[10, 15],
		[12, 15],
		[14, 15],
		[16, 15],
		[18, 15],

	])



coord1 = real_coord1
coord2 = real_coord2

"""
"""

"""
canvas1 = np.ones(shape=(28, 28))
canvas2 = np.ones(shape=(28, 28))
canvas1 = np.negative(canvas1)
canvas2 = np.negative(canvas2)
"""

canvas1 = np.zeros(shape=(28, 28))
canvas2 = np.zeros(shape=(28, 28))



for i in range(len(coord1)):
	canvas1[coord1[i][0]][coord1[i][1]] = 1

for i in range(len(coord2)):
	canvas2[coord2[i][0]][coord2[i][1]] = 1




"""
fig, axs = plt.subplots(2)
fig.suptitle("Image - Plot")
axs[0].imshow(canvas1)
axs[1].imshow(canvas2)
plt.show()

"""
#isTraining = True
isTraining = False
training_size = 25
path_to_file1 = "_saved/save_25_lvl1"
path_to_file2 = "_saved/save_25_lvl2"
path_to_file3 = "_saved/save_25_lvl3"



if isTraining:
	rules = []
	model1 = np.zeros((10, 16, 16), dtype = np.float)
	model2 = np.zeros((10, 16, 16, 16), dtype = np.float)
	model3 = np.zeros((10, 16, 16, 16, 16), dtype = np.float)
	

	for i in range(10):
		rule1 = np.zeros((16, 16), dtype = np.float)
		rule2 = np.zeros((16, 16, 16), dtype = np.float)
		rule3 = np.zeros((16, 16, 16, 16), dtype = np.float)
		idxes = check_data(i, training_size)

		size = len(idxes)
		d = prep_training_data(idxes)


		for img in d:
			btb = BTB(2, img)
			btb.perception()
			rule1 = rule1 + btb.lvl1
			rule2 = rule2 + btb.lvl2
			rule3 = rule3 + btb.lvl3

		rule1 = rule1 / size
		rule2 = rule2 / size
		rule3 = rule3 / size

		model1[i, :, :] = rule1
		model2[i, :, :, :] = rule2
		model3[i, :, :, :, :] = rule3


	"""
	for i in range(10):
		print(i)
		for j in range(10):
			print(j)
			if i == j:
				toSave[i, j, :, :, :] = rules[i]
			else:
				toSave[i, j, :, :, :] = abs(rules[i] - rules[j])
	"""

	
	np.save(path_to_file1, model1)
	np.save(path_to_file2, model2)
	np.save(path_to_file3, model3)

	#np.save(path_to_file, toSave_try)
"""
else:
	# ---------------------------- testing
	model = np.load('_saved/save_100.npy')

	#isRandom = False
	isRandom = True
	correctness = 0

	testing_size = 10

	idxes = []



	testing_data = []
	testing_labels = []


	if isRandom:
		for i in range(testing_size):
			data_size = len(labels_test) - 1
			idx_pick = random.randint(0, data_size)
			testing_data.append(images_test[idx_pick])
			testing_labels.append(labels_test[idx_pick])

	else:
		idxes = []

		for i in range(10):
			idx = check_data(i, testing_size)
			idxes = idxes + idx

		for i in idxes:
			testing_data.append(images_test[i])
			testing_labels.append(labels_test[i])

	d = prep_testing_data(testing_data)

	for i in range(len(d)):
		btb = BTB(2, i)
		btb.perception()
		r = btb.lvl2











		result1 = np.multiply(np.multiply(r, rule1), diff)
		result1 = result1.sum()

		result2 = np.multiply(np.multiply(r, rule2), diff)
		result2 = result2.sum()


		print(result1)
		print(result2)

		if result1 < result2:
			print("Right!")
			correctness = correctness + 1
"""




"""
	for idx in idxes:
		# rule1 = 7, rule2 = 3
		i = [idx]

		d = prep_training_data(i)[0]

		#plt.imshow(d)
		#plt.show()

		btb = BTB(2, d)
		btb.perception()
		r = btb.lvl2

		result1 = np.multiply(np.multiply(r, rule1), diff)
		result1 = result1.sum()

		result2 = np.multiply(np.multiply(r, rule2), diff)
		result2 = result2.sum()


		print(result1)
		print(result2)

		if result1 < result2:
			print("Right!")
			correctness = correctness + 1

	print(float(correctness) / float(num_pick))
"""

























#model = np.load('_saved/save_100.npy')
m1 = np.load(path_to_file1 + ".npy")
m2 = np.load(path_to_file2 + ".npy")
m3 = np.load(path_to_file3 + ".npy")





# -------------------------------------- test
"""
print(m1.shape)


f = np.zeros((10, 16*16), dtype = np.float)

for i in range(10):
	f[i, :] = m1[i].astype(int).flatten() 

for i in range(f.shape[1]):
	s = f[:, i].sum()

	if s == 1:
		print("--------------")
		print(1)
		print(i)
		print(f[:, i].tolist().index(1))
		print("--------------")
	if s == 9:
		print("--------------")
		print(0)
		print(i)
		print(f[:, i].tolist().index(0))
		print("--------------")
"""



# -------------------------------------- targeting test
bias = 0.1

"""
l1 + l2:
[0.8, 0.96, 0.3, 0.6, 0.44, 0.44, 0.38, 0.56, 0.22, 0.32]
0.502

l1 + l2 + l3:
training: 25*10
testing: 500
[0.7, 0.96, 0.38, 0.7, 0.68, 0.78, 0.7, 0.74, 0.6, 0.6]
0.684


l1 + l2 + l3:
training: 25*10
testing: 500
with weights
[0.66, 0.96, 0.46, 0.72, 0.68, 0.72, 0.6, 0.74, 0.7, 0.6]
0.684
"""

correctness = 0
testing_size = 50
s = 10 * testing_size

accurate_rate = []


for i in range(10):
	local_corresness = 0
	idxes = get_testing_data_labeled(i, testing_size)

	size = len(idxes)
	d = prep_testing_data_labeled(idxes)

	for img in d:
		btb = BTB(2, img)
		btb.perception()
		r1 = btb.lvl1
		r2 = btb.lvl2
		r3 = btb.lvl3

		res1 = []
		res2 = []
		res3 = []

		r1 = r1.flatten()
		r2 = r2.flatten()
		r3 = r3.flatten()

		for k in range(10):
			e1 = m1[k].flatten()
			ed1 = np.linalg.norm(e1 - r1)
			res1.append(ed1)

			e2 = m2[k].flatten()
			ed2 = np.linalg.norm(e2 - r2)
			res2.append(ed2)

			e3 = m3[k].flatten()
			ed3 = np.linalg.norm(e3 - r3)
			res3.append(ed3)



		reorder1 = sorted(range(len(res1)), key=lambda k: res1[k])
		reorder2 = sorted(range(len(res2)), key=lambda k: res2[k])
		reorder3 = sorted(range(len(res3)), key=lambda k: res3[k])



		# ---------------------- try (1)
		r1_w = 1.5
		r2_w = 1.3
		r3_w = 1.1

		# ---------------------- try (1)



		

		result = []

		for c in range(10):
			rank_l1 = float(reorder1.index(c)) * r1_w
			rank_l2 = float(reorder2.index(c)) * r2_w
			rank_l3 = float(reorder3.index(c)) * r3_w
			result.append(rank_l1 + rank_l2 + rank_l3)















		"""
		pred = result.index(min(result))

		print(str(i) + " - " + str(pred))
		print(sorted(range(len(result)), key=lambda k: result[k]))

		if i == pred:
			correctness = correctness + 1
			local_corresness = local_corresness + 1


		"""

		res_min = min(result)

		pred = [i for i, x in enumerate(result) if x == res_min]
		print(str(i) + " - " + str(pred))
		print(sorted(range(len(result)), key=lambda k: result[k]))

		if i in pred:
			correctness = correctness + 1
			local_corresness = local_corresness + 1















	accurate_rate.append(float(local_corresness) / float(testing_size))


print(accurate_rate)
print(float(correctness) / float(s))










"""
# -------------------------------------- random test

correctness = 0
testing_size = 500


for i in range(testing_size):
	print("--------------> " + str(i))
	data_size = len(labels_test) - 1
	idx_pick = random.randint(0, data_size)
	label = labels_test[idx_pick]

	idx = [idx_pick]

	d = prep_training_data(idx)[0]

	btb = BTB(2, d)
	btb.perception()

	r1 = btb.lvl1
	r2 = btb.lvl2
	r3 = btb.lvl3

	res1 = []
	res2 = []
	res3 = []

	r1 = r1.flatten()
	r2 = r2.flatten()
	r3 = r3.flatten()

	for k in range(10):
		e1 = m1[k].flatten()
		ed1 = np.linalg.norm(e1 - r1)
		res1.append(ed1)

		e2 = m2[k].flatten()
		ed2 = np.linalg.norm(e2 - r2)
		res2.append(ed2)

		e3 = m3[k].flatten()
		ed3 = np.linalg.norm(e3 - r3)
		res3.append(ed3)

	reorder1 = sorted(range(len(res1)), key=lambda k: res1[k])
	reorder2 = sorted(range(len(res2)), key=lambda k: res2[k])
	reorder3 = sorted(range(len(res3)), key=lambda k: res3[k])

	result = []

	for c in range(10):
		rank_l1 = reorder1.index(c)
		rank_l2 = reorder2.index(c)
		rank_l3 = reorder3.index(c)
		result.append(rank_l1 + rank_l2 + rank_l3)

	res_min = min(result)

	pred = [i for i, x in enumerate(result) if x == res_min]
	print(str(i) + " - " + str(pred))
	print(sorted(range(len(result)), key=lambda k: result[k]))

	if i in pred:
		correctness = correctness + 1


print(float(correctness) / float(testing_size))

"""












"""
for i in range(testing_size):
	print("--------------> " + str(i))
	data_size = len(labels_test) - 1
	idx_pick = random.randint(0, data_size)
	label = labels_test[idx_pick]

	idx = [idx_pick]

	d = prep_training_data(idx)[0]

	btb = BTB(2, d)
	btb.perception()

	r1 = btb.lvl1

	res1 = []

	r1 = r1.flatten()

	for k in range(10):
		e1 = m1[k].flatten()
		ed1 = np.linalg.norm(e1 - r1)
		res1.append(ed1)



	pred = res1.index(min(res1))

	print(str(label) + " - " + str(pred))
	print(sorted(range(len(res1)), key=lambda k: res1[k]))
	print(sorted(res1))

	if label == pred:
		correctness = correctness + 1


print(float(correctness) / float(testing_size))

# -------------------------------------- test
"""





















"""
correctness = 0
testing_size = 10


a = []
for i in range(10):
	p = m1[i].astype(int).flatten()
	a.append(p)

fig, axs = plt.subplots(10)
axs[0].plot(a[0])
axs[1].plot(a[1])
axs[2].plot(a[2])
axs[3].plot(a[3])
axs[4].plot(a[4])
axs[5].plot(a[5])
axs[6].plot(a[6])
axs[7].plot(a[7])
axs[8].plot(a[8])
axs[9].plot(a[9])

plt.show()









p1 = 0
p2 = 1

_11 = m1[p1].flatten()
_12 = m2[p1].flatten()
_13 = m3[p1].flatten()

_21 = m1[p2].flatten()
_22 = m2[p2].flatten()
_23 = m3[p2].flatten()

fig, axs = plt.subplots(6)
axs[0].plot(_11)
axs[1].plot(_12)
axs[2].plot(_13)

axs[3].plot(_21)
axs[4].plot(_22)
axs[5].plot(_23)

plt.show()
"""



























"""
# -------------------------------------- test
bias = 0.1


#for k in range(10):
#	orig[k] = orig[k][:, :, :] + bias
#	orig[k] = orig[k].astype(int)



correctness = 0
testing_size = 10

for i in range(testing_size):
	print("--------------> " + str(i))
	data_size = len(labels_test) - 1
	idx_pick = random.randint(0, data_size)
	label = labels_test[idx_pick]

	idx = [idx_pick]

	d = prep_training_data(idx)[0]

	btb = BTB(2, d)
	btb.perception()


	r1 = btb.lvl1
	r2 = btb.lvl2
	r3 = btb.lvl3

	res1 = []
	res2 = []
	res3 = []

	total = []


	for k in range(10):
		e1 = m1[k].flatten()
		r1 = r1.flatten()
		ed1 = np.linalg.norm(e1 - r1)
		res1.append(ed1)

		e2 = m2[k].flatten()
		r2 = r2.flatten()
		ed2 = np.linalg.norm(e2 - r2)
		res2.append(ed2)

		e3 = m3[k].flatten()
		r3 = r3.flatten()
		ed3 = np.linalg.norm(e3 - r3)
		res3.append(ed3)

		total.append(ed1 + ed2 + ed3)











	pred = total.index(min(total))

	print(str(label) + " - " + str(pred))
	print(sorted(range(len(total)), key=lambda k: total[k]))
	print(sorted(total))

	if label == pred:
		correctness = correctness + 1


print(float(correctness) / float(testing_size))

# -------------------------------------- test
"""




































"""
for i in range(10):
	print("---------: " + str(i))
	for j in range(10):
		print(str(j) + ": " + str(model[i, j, :, :, :].sum()))
"""


"""
for i in range(10):
	rule_a = model[i, i, :, :, :]

	result_a = np.multiply(r, rule_a)
	result_a = result_a.sum()

	res.append(result_a)
"""

"""
for i in range(10):
	temp = 0
	for j in range(10):
		a = i
		b = j

		rule_a = model[a, a, :, :, :]
		rule_b = model[b, b, :, :, :]
		diff = model[a, b, :, :, :]

		result_a = np.multiply(np.multiply(r, rule_a), diff)
		result_a = result_a.sum()

		result_b = np.multiply(np.multiply(r, rule_b), diff)
		result_b = result_b.sum()


		if result_a >= result_b:
			temp = temp + 1

	res.append(temp)
"""

#print(res)
#print(res.index(max(res)))
























