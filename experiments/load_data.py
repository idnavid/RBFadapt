import numpy as np
import matplotlib.pyplot as plt
# Collection of methods used to access real-world regression data for experiments. 

def normalize(X,low=0,high=1):
	"""
	Normalize X across each column. 
	"""
	(N,d) = X.shape
	X_mins = np.min(X,axis=0)
	X_maxs = np.max(X,axis=0)
	# NOTE: I had to move X_range = X_maxs - X_mins
	# before the if conditions, because broadcasting 
	# affected X_maxs and X_mins. 
	X_range = X_maxs - X_mins
	if low == 0:
		X = X - X_mins[np.newaxis,:]
	elif low ==-1:
		X_means = np.mean(X,axis=0)
		X = X - X_means[np.newaxis,:]
	X = np.divide(X,1e-8+(X_range[np.newaxis,:]))
	return X
	

######################################################################
def abalone(mode='train'):
	filename = 'C:/Users/nshokouhi/Downloads/RBF_project/data/abalone/data.txt'
	fin = open(filename)
	gender_map = {'M':0.,'F':1.,'I':2.}
	attr = {'gender' :[],
			'length' :[],
			'diam'   :[],
			'height' :[],
			'whole'  :[],
			'shucked':[],
			'viscera':[],
			'shell'  :[],
			'rings'  :[]}

	for i in fin:
		line = i.strip().split(',')
		attr['gender'].append(gender_map[line[0]])
		attr['length'].append(float(line[1]))
		attr['diam'].append(float(line[2]))
		attr['height'].append(float(line[3]))
		attr['whole'].append(float(line[4]))      
		attr['shucked'].append(float(line[5]))
		attr['viscera'].append(float(line[6]))
		attr['shell'].append(float(line[7]))
		attr['rings'].append(float(line[8]))

	fin.close()
	N = len(attr['gender'])
	d = len(attr)-1
	I = np.zeros((N,d))
	
	I[:,0] = np.array(attr['gender'])
	I[:,1] = np.array(attr['length'])
	I[:,2] = np.array(attr['diam'])
	I[:,3] = np.array(attr['height'])
	I[:,4] = np.array(attr['whole'])
	I[:,5] = np.array(attr['shucked'])
	I[:,6] = np.array(attr['viscera'])
	I[:,7] = np.array(attr['shell'])
	O = np.zeros((N,1))
	O[:,0] = np.array(attr['rings'])

	I = normalize(I)
	O = normalize(O,low=-1,high=1)
	if mode=='train':
		return I[:2000,:],O[:2000]
	elif mode=='test':
		return I[2000:,:],O[2000:] 
	return I,O

#######################################################################
def ailerons(mode='train'):
	raise NameError('Ailerone is too large! I put this error here in case I forget.')
	exit()

	# Read attributes:
	# This data has about 40 attributes, so it's easier
	# to read them from file. 
	attr = {}
	attr_map = {} # maps attribute names to indexes
	attr_file = 'C:/Users/nshokouhi/Downloads/RBF_project/data/Ailerons/ailerons.domain'
	fin = open(attr_file)
	j = 0
	for i in fin:
		key = i.strip().split(':')[0]
		attr[key] = []
		attr_map[j] = key
		j+=1
	fin.close()
	if mode=='train':
		filename = 'C:/Users/nshokouhi/Downloads/RBF_project/data/Ailerons/ailerons.data'
	elif mode=='test':
		filename = 'C:/Users/nshokouhi/Downloads/RBF_project/data/Ailerons/ailerons.test'

	fin = open(filename)
	for i in fin:
		line = i.strip().split(', ')
		for j in range(len(line)):
			attr[attr_map[j]].append(float(line[j]))
	fin.close()
	
	N = len(attr[attr_map[0]])
	d = len(attr)-1 # Don't include the goal
	I = np.zeros((N,d))
	
	for i in range(d):
		I[:,i] = np.array(attr[attr_map[i]])

	O = np.zeros((N,1))
	O[:,0] = np.array(attr['goal'])
	
	I = normalize(I)
	O = normalize(O,low=-1,high=1)
	return I,O


#######################################################################
def boston_housing(mode='train'):
	data_file = 'C:/Users/nshokouhi/Downloads/RBF_project/data/boston_housing/HO.dat'
	fin = open(data_file)

	attr = {}
	goal = []
	for i in fin:
		line_list = i.strip().split(' ')
		attr_idx = 0
		for j in line_list[:-1]:
			line_value = j.strip()
			if line_value!='':
				if attr_idx in attr:
					attr[attr_idx].append(float(line_value))
				else:
					attr[attr_idx] =[float(line_value)]
				attr_idx +=1
		goal.append(float(line_list[-1]))
	fin.close()
	
	N = len(goal)
	d = len(attr)
	I = np.zeros((N,d))
	for i in range(d):
		I[:,i] = np.array(attr[i])
	O = np.zeros((N,1))
	O[:,0] = np.array(goal)
	N_train = 354
	if mode=='train':
		I = I[:N_train,:]
		O =O[:N_train]
		I = normalize(I)
		O = normalize(O,low=-1,high=1)
		return I,O
	elif mode=='test':
		I = I[N_train:,:]
		O =O[N_train:]
		I = normalize(I)
		O = normalize(O,low=-1,high=1)
		return I,O
	return I,O


#######################################################################
def weather_ankara(mode='train'):
	data_file = 'C:/Users/nshokouhi/Downloads/RBF_project/data/weather_ankara/WA.dat'
	fin = open(data_file)

	attr = {}
	goal = []
	for i in fin:
		line_list = i.strip().split(' ')
		attr_idx = 0
		for j in line_list[:-1]:
			line_value = j.strip()
			if line_value!='':
				if attr_idx in attr:
					attr[attr_idx].append(float(line_value))
				else:
					attr[attr_idx] =[float(line_value)]
				attr_idx +=1
		goal.append(float(line_list[-1]))
	fin.close()
	
	N = len(goal)
	d = len(attr)
	I = np.zeros((N,d))
	for i in range(d):
		I[:,i] = np.array(attr[i])
	O = np.zeros((N,1))
	O[:,0] = np.array(goal)
	N_train = 1100
	if mode=='train':
		I = I[:N_train,:]
		O =O[:N_train]
		I = normalize(I)
		O = normalize(O,low=-1,high=1)
		return I,O
	elif mode=='test':
		I = I[N_train:,:]
		O =O[N_train:]
		I = normalize(I)
		O = normalize(O,low=-1,high=1)
		return I,O
	return I,O

#######################################################################
def auto_mpg(mode='train'):
	data_file = 'C:/Users/nshokouhi/Downloads/RBF_project/data/auto_mpg/tmp.data'
	fin = open(data_file)

	attr = {}
	goal = []
	for i in fin:
		raw = i.strip().split('"')[0] # everything after " is the car brand
		line_list_raw = raw.strip().split(' ')
		# line_list_raw has car names in it. That's why we need 
		# to strip off all the non-float values. 
		# Also, ? means there's a missing value. I set those to 0. 
		line_list = []
		for i in line_list_raw:
			i = i.strip()
			if i=='?':
				i = 0.
				line_list.append(i)
			elif i=='':
				pass
			elif '\t' in i:
				line_list.append(float(i.split('\t')[0]))
			else:
				try:
					line_list.append(float(i))
				except:
					pass
		attr_idx = 0			
		for j in line_list[:-1]:	
			line_value = j	
			if attr_idx in attr:
				attr[attr_idx].append(line_value)
			else:
				attr[attr_idx] =[line_value]
			attr_idx +=1
		if line_list[-1]>3.:
			print (line_list_raw)
			exit()
		goal.append(float(line_list[-1]))

	fin.close()

	N = len(goal)
	d = len(attr)
	I = np.zeros((N,d))
	for i in range(d):
		I[:,i] = np.array(attr[i])
	O = np.zeros((N,1))
	O[:,0] = np.array(goal)
	I = normalize(I)
	O = normalize(O,low=-1,high=1)

	N_train = 320
	if mode=='train':
		I = I[:N_train,:]
		O =O[:N_train]
		return I,O
	elif mode=='test':
		I = I[N_train:,:]
		O =O[N_train:]
		return I,O
	return I,O


if __name__=='__main__':
	I,O = weather_ankara('train')
	print(I.shape,O.shape)
	I,O = weather_ankara('test')
	print(I.shape,O.shape)