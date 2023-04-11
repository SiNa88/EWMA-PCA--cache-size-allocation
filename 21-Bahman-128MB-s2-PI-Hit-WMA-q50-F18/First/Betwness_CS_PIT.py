from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import FactorAnalysis


import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
from scipy.misc import factorial

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pylab import *

def ExpMovingAvg(values, window):
	#print values.shape
	weights = np.exp(np.linspace(-1.,0.,window))
	weights /= weights.sum()
	#print weights.shape
	a = np.convolve(values,weights,mode='full') [:len(values)]
	#print a
	return a

def weightedmovingaverage(values, window):
	
	#print float(window_size)
	weights = np.linspace(1,window,window)
	#print weights
	weights /= weights.sum()
	#print weights
	return np.convolve(values, weights, 'same')

def weightedmovingaverage2(values, window):
	
	#print float(window_size)
	weights = np.linspace(1,window,window)
	#print weights
	weights /= weights.sum()
	#print weights
	return np.mul(values, weights)

def movingaverage(values, window):
	weights = np.ones(int(window))/float(window)
	#print weights
	return np.convolve(values, weights, 'same')

def My_PCA(X):
	#print X
	X_std = StandardScaler().fit_transform(X)

	######max-min_Normalizing######
	min_X_std = np.min(X_std,axis=0)
	X_std_new = zeros((11,3))
	#print min_X_std.item(0)
	for i in range(X_std.shape[0]):
		for j in range(X_std.shape[1]):
 			X_std_new[i,j] = (X_std[i,j]+((-1)*min_X_std.item(j)))
	#print X_std_new
	X_std_Norm = np.zeros((11,3))
	max_X_std_new = np.max(X_std_new,axis=0)
	#print max_X_std_new
	for i in range(X_std.shape[0]):
		for j in range(X_std.shape[1]):
 			X_std_Norm[i,j] = X_std_new[i,j]/max_X_std_new.item(j)
	print X_std_Norm
	##############################

	data = np.zeros((33,1))
	data[0:11,0] = X_std_Norm[:,0]
	data[11:22,0] = X_std_Norm[:,1]
	data[22:33,0] = X_std_Norm[:,2]
	MyPlot2DimReshaped(data)

	sklearn_pca = PCA(n_components=1)
	sklearn_transf = sklearn_pca.fit_transform(X_std)
	#print sklearn_transf
	sklearn_pca2D = PCA(n_components=3)
	sklearn_transf2D = sklearn_pca2D.fit_transform(X_std)
	#print sklearn_transf2D.shape
	min_sklearn_transf = np.min(sklearn_transf,axis=0)
	sklearn_transf_new = (sklearn_transf+((-1)*min_sklearn_transf))
	#print sklearn_transf_new
	sklearn_transf_Norm = sklearn_transf_new/np.sum(sklearn_transf_new)
	#print sklearn_transf_Norm
	sklearn_transf2 = np.around(sklearn_transf_Norm*143000,0)
	MyPCAPlot(sklearn_transf2)
	print sklearn_transf2
	return sklearn_transf2

def My_KernelPCA(X):
	X_std = StandardScaler().fit_transform(X)
	
	######max-min_Normalizing######
	min_X_std = np.min(X_std,axis=0)
	X_std_new = zeros((11,3))
	#print min_X_std.item(0)
	for i in range(X_std.shape[0]):
		for j in range(X_std.shape[1]):
 			X_std_new[i,j] = (X_std[i,j]+((-1)*min_X_std.item(j)))
	#print X_std_new
	X_std_Norm = np.zeros((11,3))
	max_X_std_new = np.max(X_std_new,axis=0)
	#print max_X_std_new
	for i in range(X_std.shape[0]):
		for j in range(X_std.shape[1]):
 			X_std_Norm[i,j] = X_std_new[i,j]/max_X_std_new.item(j)
	print X_std_Norm
	##############################
	#print ":D :D :D" 

	sklearn_pca = KernelPCA(n_components=1)
	sklearn_transf = sklearn_pca.fit_transform(X_std)
	#sklearn_transf = (-1)*sklearn_transf
	min_sklearn_transf = np.min(sklearn_transf,axis=0)
	sklearn_transf_new = (sklearn_transf+((-1)*min_sklearn_transf))	
	#print sklearn_transf	
	sum_sklearn_transf = np.sum(sklearn_transf_new)
	#print sum_sklearn_transf
	sklearn_transf_Norm = sklearn_transf_new/sum_sklearn_transf
	#print sklearn_transf_Norm
	sklearn_transf2 = np.around(sklearn_transf_Norm*143000,0)
	MyPCAPlot(sklearn_transf2)
	print sklearn_transf2
	return sklearn_transf2

def My_SparsePCA_TOTAL(X):
	X_std = StandardScaler().fit_transform(X)
	sklearn_pca = SparsePCA(n_components=1)
	sklearn_transf = sklearn_pca.fit_transform(X_std)
	sklearn_transf = np.abs(sklearn_transf)
	sklearn_transf = (-1)*sklearn_transf
	min_sklearn_transf = np.min(sklearn_transf,axis=0)
	sklearn_transf_new = (sklearn_transf+((-1)*min_sklearn_transf))	
	#print sklearn_transf	
	sum_sklearn_transf = np.sum(sklearn_transf_new)
	#print sum_sklearn_transf
	sklearn_transf_Norm = sklearn_transf_new/sum_sklearn_transf
	#print sklearn_transf_Norm
	sklearn_transf2 = np.around(sklearn_transf_Norm*143000,0)
	MyPCAPlot(sklearn_transf2)
	print sklearn_transf2
	return sklearn_transf2


def My_FactorAnalysis(X):
	X_std = StandardScaler().fit_transform(X)
	sklearn_pca = FactorAnalysis(n_components=1)
	sklearn_transf = sklearn_pca.fit_transform(X_std)
	sklearn_transf = np.abs(sklearn_transf)
	#min_sklearn_transf = np.min(sklearn_transf)
	#print sklearn_transf
	#print min_sklearn_transf
	#sklearn_transf = (sklearn_transf+((-1)*min_sklearn_transf))
	sum_sklearn_transf = np.sum(sklearn_transf)
	sklearn_transf = (sklearn_transf)/sum_sklearn_transf
	sklearn_transf2 = np.around(sklearn_transf*143000,0)
	#print sklearn_transf
	return sklearn_transf2

def My_Isomap(X):
	n_neighbors = 1
	n_components = 1
	sklearn_transf = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
	sklearn_transf = np.abs(sklearn_transf)
	#min_sklearn_transf = np.min(sklearn_transf)
	#print sklearn_transf
	#print min_sklearn_transf
	#sklearn_transf = (sklearn_transf+((-1)*min_sklearn_transf))
	sum_sklearn_transf = np.sum(sklearn_transf)
	sklearn_transf = (sklearn_transf)/sum_sklearn_transf
	sklearn_transf2 = np.around(sklearn_transf*143000,0)
	#print sklearn_transf
	return sklearn_transf2

def MyStem2(X1):
	X1_1stRow = X1[22,:].transpose()
	x_axis=np.linspace(1,600,600)
	plt.scatter(x_axis,X1_1stRow)
	plt.plot(x_axis,X1_1stRow)
	a = (1/(x_axis))
	plt.plot(x_axis,a)
	plt.xlim([0,600])
	plt.ylim([0,25])
	plt.xlabel('Time(sec)')
	plt.ylabel('Recieved_Requests')
	#plt.ylabel('Traffic')
	#plt.title('Traffic_Preformance_of_Rtr6')
	plt.title('PendingInterestTable_Preformance_of_Rtr10')
	plt.show()

def PIT():
	alfa = 0.125 ;
	beta = 0.25 ;

	'''m1 = np.loadtxt("Rcvd-Requests-Homo-PerSec-1.csv", delimiter=',')
	X1=np.matrix(m1)
	m2 = np.loadtxt("Rcvd-Requests-Homo-PerSec-2.csv", delimiter=',')
	X2=np.matrix(m2)
	m3 = np.loadtxt("Rcvd-Requests-Homo-PerSec-3.csv", delimiter=',')
	X3=np.matrix(m3)
	m4 = np.loadtxt("Rcvd-Requests-Homo-PerSec-4.csv", delimiter=',')
	X4=np.matrix(m4)'''
	m5 = np.loadtxt("Rcvd-Requests-Homo-PerSec-5.csv", delimiter=',')
	X5=np.matrix(m5)
	'''m6 = np.loadtxt("Rcvd-Requests-Homo-PerSec-6.csv", delimiter=',')
	X6=np.matrix(m6)
	m7 = np.loadtxt("Rcvd-Requests-Homo-PerSec-7.csv", delimiter=',')
	X7=np.matrix(m7)
	m8 = np.loadtxt("Rcvd-Requests-Homo-PerSec-8.csv", delimiter=',')
	X8=np.matrix(m8)
	m9 = np.loadtxt("Rcvd-Requests-Homo-PerSec-9.csv", delimiter=',')
	X9=np.matrix(m9)
	m10 = np.loadtxt("Rcvd-Requests-Homo-PerSec-10.csv", delimiter=',')
	X10=np.matrix(m10)'''
	m_mean = np.zeros((11,600))
	EMA_mean=np.zeros((11,600))
	WMA_mean = np.zeros((11,600))
	MyEstimation = np.zeros((11))
	MyDeviation = np.zeros((11))
	MyPITImportance = np.zeros((11))
	
	min = 1000
	for i in range(m_mean.shape[0]):
		for j in range(m_mean.shape[1]):
			#print i
			#X5
			m_mean.itemset((i,j),(X5.item(i,j)))
			#+X2.item(i,j)+X3.item(i,j)+X4.item(i,j)+X5.item(i,j)+X6.item(i,j)+X7.item(i,j)+X8.item(i,j)+X9.item(i,j)+X10.item(i,j))/10)
		#WMA_mean[i,0:10] = m_mean[i,0:10]
		#print WMA_mean[i,0:10]
		WMA_mean[i,:] = weightedmovingaverage(m_mean[i,:], 10)
		'''WMA_mean[i,0:10] = weightedmovingaverage(m_mean[i,0:10], 10)
		WMA_mean[i,0:9] = weightedmovingaverage(m_mean[i,0:9], 9)
		WMA_mean[i,0:8] = weightedmovingaverage(m_mean[i,0:8], 8)
		WMA_mean[i,0:7] = weightedmovingaverage(m_mean[i,0:7], 7)
		WMA_mean[i,0:6] = weightedmovingaverage(m_mean[i,0:6], 6)
		WMA_mean[i,0:5] = weightedmovingaverage(m_mean[i,0:5], 5)
		WMA_mean[i,0:4] = weightedmovingaverage(m_mean[i,0:4], 4)
		WMA_mean[i,0:3] = weightedmovingaverage(m_mean[i,0:3], 3)
		WMA_mean[i,0:2] = weightedmovingaverage(m_mean[i,0:2], 2)'''
		WMA_mean[i,0:10] = m_mean[i,0:10]
		####### Averaging on All the observations to get a number for each Router #######
		MyPITImportance.itemset((i),np.mean(WMA_mean[i,:], axis=0))

		if MyPITImportance[i] < min and MyPITImportance[i]!=0:
			min = MyPITImportance[i]
	font = {'family' : 'STIXGeneral',
        	'weight' : 'bold',
        	'size'   : 32}

	matplotlib.rc('font', **font)
	x = np.linspace(0,600,600)

	###### Averaging on All Core Routers ######
	#m_mean_Avg = np.zeros((1,600))
	m_mean_Avg = np.average(m_mean,axis = 0)
	WMA_mean_Avg = np.average(WMA_mean,axis = 0)
	###########################################
	Error  = np.abs(WMA_mean_Avg-m_mean_Avg)
	print np.average(Error)

	'''hist, bin_edges = np.histogram(m_mean_Avg, bins = range(50))
	print bin_edges
	plt.bar(bin_edges[:-1], hist, width=1)
	plt.xlim(np.min(bin_edges), np.max(bin_edges))
	plt.title('Histogram of Pending Interests in an NDN Router')
	#plt.ylim([0,40])
	plt.show()'''

	##patterns = ('x')
	
	# the bins should be of integer width, because poisson is an integer distribution
	entries, bin_edges, patches = plt.hist(m_mean_Avg, bins=9, range=[-0.5, 10.5], normed=True, color='white', hatch='x', label='Simulation')
	
	# calculate binmiddles
	bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])

	# poisson function, parameter lamb is the fit parameter
	def poisson(k, lamb):
	    return (lamb**k/factorial(k)) * np.exp(-lamb)

	# fit with curve_fit
	parameters, cov_matrix = curve_fit(poisson, bin_middles, entries) 

	# plot poisson-deviation with fitted parameter
	x_plot = np.linspace(0, 20, 1000)
	plt.xlim([-1,20])
	plt.title('Histogram of Pending Interests in an NDN Router')
	plt.plot(x_plot, poisson(x_plot, *parameters), 'r-', lw=2,label='Theory')
	plt.legend()
	plt.show()

	for i in range(m_mean.shape[0]):
		if MyPITImportance[i] == 0:
			MyPITImportance[i] = min
	return MyPITImportance
	
def CS():
	alfa = 0.125 ;
	beta = 0.25 ;
	
	m1 = np.loadtxt("Homo-Hit_PerSec-1.csv", delimiter=',')
	X1=np.matrix(m1)
	'''m2 = np.loadtxt("Homo-Hit_PerSec-2.csv", delimiter=',')
	X2=np.matrix(m2)
	m3 = np.loadtxt("Homo-Hit_PerSec-3.csv", delimiter=',')
	X3=np.matrix(m3)
	m4 = np.loadtxt("Homo-Hit_PerSec-4.csv", delimiter=',')
	X4=np.matrix(m4)
	m5 = np.loadtxt("Homo-Hit_PerSec-5.csv", delimiter=',')
	X5=np.matrix(m5)
	m6 = np.loadtxt("Homo-Hit_PerSec-6.csv", delimiter=',')
	X6=np.matrix(m6)
	m7 = np.loadtxt("Homo-Hit_PerSec-7.csv", delimiter=',')
	X7=np.matrix(m7)
	m8 = np.loadtxt("Homo-Hit_PerSec-8.csv", delimiter=',')
	X8=np.matrix(m8)
	m9 = np.loadtxt("Homo-Hit_PerSec-9.csv", delimiter=',')
	X9=np.matrix(m9)
	m10 = np.loadtxt("Homo-Hit_PerSec-10.csv", delimiter=',')
	X10=np.matrix(m10)'''
	m_mean = np.zeros((11,600))
	WMA_mean = np.zeros((11,600))
	EMA_mean = np.zeros((11,600))
	MyEstimation = np.zeros((11))
	MyDeviation = np.zeros((11))
	MycsImportance = np.zeros((11))
	
	min = 1000
	for i in range(m_mean.shape[0]):
		for j in range(m_mean.shape[1]):

			m_mean.itemset((i,j),(X1.item(i,j)))
			#+X2.item(i,j)+X3.item(i,j)+X4.item(i,j)+X5.item(i,j)+X6.item(i,j)+X7.item(i,j)+X8.item(i,j)+X9.item(i,j)+X10.item(i,j))/10)
		WMA_mean[i,:] = weightedmovingaverage(m_mean[i,:], 10)
		WMA_mean[i,0:10] = m_mean[i,0:10]
		
		####### Averaging on All the observations to get a number for each Router #######
		MycsImportance.itemset((i),np.mean(WMA_mean[i,:], axis=0))
		#print MycsImportance[i]

		if MycsImportance[i] < min and MycsImportance[i]!=0:
			min = MycsImportance[i]
	font = {'family' : 'STIXGeneral',
        	'weight' : 'bold',
        	'size'   : 32}

	matplotlib.rc('font', **font)
	x = np.linspace(0,600,600)

	###### Averaging on All Core Routers ######
	#m_mean_Avg = np.zeros((1,600))
	m_mean_Avg = np.average(m_mean,axis = 0)
	WMA_mean_Avg = np.average(WMA_mean,axis = 0)
	###########################################

	hist, bin_edges = np.histogram(m_mean_Avg, bins = range(20))
	plt.bar(bin_edges[:-1], hist, width=1)
	plt.xlim(np.min(bin_edges), np.max(bin_edges))
	plt.title('Histogram of Interest Hits in an NDN Core Router')
	plt.show() 

	Error  = np.abs(WMA_mean_Avg-m_mean_Avg)
	print np.average(Error)

	#print WMA_mean_Avg.shape
	matplotlib.rc('font', **font)
	x = np.linspace(0,600,600)
	plt.plot(x,m_mean_Avg,'.-b',label="Observations")
	plt.plot(x,WMA_mean_Avg,'o-r',label="WMA of Observations")
	plt.xlabel('Simulation Time(Seconds)')
	plt.ylabel('Hit#')
	plt.legend(loc='lower center')
	plt.xlim([0,600])
	plt.ylim([0,20])
	plt.title('Number of Hits in an NDN Core Router')
	plt.show()
	for i in range(m_mean.shape[0]):
		if MycsImportance[i] == 0:
			MycsImportance[i] = min
	MycsImportance = MycsImportance/sum(MycsImportance)
	#print MycsImportance
	return MycsImportance

def PIT_CS_Betwness(pit,cs,Betwness,load):

	#min_pit = pit.min()
	#max_pit = pit.max()
	sum_pit = np.sum(pit)
	pit_norm = np.zeros((11))
	#min_cs = cs.min()
	#max_cs = cs.max()
	sum_cs = np.sum(cs)
	cs_norm = np.zeros((11))
	#min_Betwness = Betwness.min()
	#max_Betwness = Betwness.max()
	sum_Betwness = np.sum(Betwness)
	Betwness_norm = np.zeros((11))
	#min_load = load.min()
	#max_load = load.max()
	sum_load = np.sum(load)
	load_norm = np.zeros((11))
	final = np.zeros((11,3))
	for i in range(cs.shape[0]):
		Betwness_norm.itemset((i),(Betwness.item(i)/sum_Betwness))
		final.itemset((i,0),Betwness_norm.item(i))
		#load_norm.itemset((i),(load.item(i)/sum_load))
		#final.itemset((i,1),load_norm.item(i))		
		pit_norm.itemset((i),(pit.item(i)/sum_pit))
		final.itemset((i,1),pit_norm.item(i))
		cs_norm.itemset((i),(cs.item(i)/sum_cs))
		final.itemset((i,2),cs_norm.item(i))		
	return final

def MyPlot(data):
	#print data
	nx, ny = data.shape[0], data.shape[1]
	x = range(0,nx,1)
	y = range(0,ny,1)
	fig = plt.figure()
	ax = Axes3D(fig)
	#X, Y = np.meshgrid(x, y)	
	X, Y = np.meshgrid(data[:,0], data[:,1])
	#print X,"\n", Y
	z_axis = sp.sparse.spdiags(data[:,2], diags, nx, ny)
	colors = np.random.rand(33)
	area = np.pi * (10 * 1)**2
	font = {'family' : 'STIXGeneral',
        	'weight' : 'bold',
        	'size'   : 28}

	matplotlib.rc('font', **font)
	ax.scatter(X,Y,z_axis,s=area, c=colors)
	ax.set_xlabel('Betweenness')
	ax.set_ylabel('PendingInterest')
	ax.set_zlabel('SatisfiedInterests')	
	plt.title('Observation(t=3)')	
	plt.show()	

def MyPlot2Dim(data):
	#data2 = data.reshape((22, 2))
	#x = np.linspace(1,data2[:,0],data2[:,0])

	colors = np.random.rand(22)
	area = np.pi * (10 * 1)**2
	font = {'family' : 'STIXGeneral',
        	'weight' : 'bold',
        	'size'   : 26}

	matplotlib.rc('font', **font)
	plt.scatter(data[:,0],data[:,1],s=area, c=colors)
	#plt.xlabel('Samples')
	#plt.ylabel('Values')
	plt.xlim([-100,np.max(data[:,0])+100])
	plt.ylim([-400,27000])
	plt.title('Observations')	
	plt.show()
	#ax.xlabel('Node')
	#plt.ylabel('Time')	
	#ax.ylabel('Metrics')

def MyPlotNotReshaped(data):
	#data2 = data.reshape((22, 2))
	x = np.linspace(1,data.shape[0],data.shape[0])
	print data.shape[0]
	colors1 = np.random.rand(1)
	colors2 = np.random.rand(1)
	colors3 = np.random.rand(1)
	area = np.pi * (10 * 1)**2
	font = {'family' : 'STIXGeneral',
        	'weight' : 'bold',
        	'size'   : 30}

	matplotlib.rc('font', **font)
	plt.scatter(x,data[:,0],s=area, c='g')
	plt.scatter(x,data[:,1],s=area, c='r')
	plt.scatter(x,data[:,2],s=area, c='b')
	#plt.xlabel('Samples')
	#plt.ylabel('Values')
	plt.xlim([0,12])
	plt.ylim([0,22500])
	plt.title('Observations')	
	plt.show()
	
def MyPlot2DimReshaped(data):
	#data2 = data.reshape((22, 2))
	x = np.linspace(1,33,33)

	colors = np.random.rand(33)
	area = np.pi * (10 * 1)**2
	font = {'family' : 'STIXGeneral',
        	'weight' : 'bold',
        	'size'   : 26}

	matplotlib.rc('font', **font)
	plt.scatter(x,data,s=area, c=colors)
	#plt.xlabel('Samples')
	#plt.ylabel('Values')
	plt.xlim([0,34])
	plt.ylim([-0.05,1.1])
	plt.title('Observations')
	plt.show()
	
def MyPCAPlot(data):
	
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			if data[i,j] >= 26000:
				data[i,j] = 26000
	# print data.shape
	data2 = data.reshape((data.shape[0]*data.shape[1], 1))
	x = np.linspace(1,data2.shape[0],data2.shape[0])
	#x = np.linspace(1,data.shape[0],data.shape[0])
	area = np.pi * (10 * 1)**2
	font = {'family' : 'STIXGeneral',
        	'weight' : 'bold',
        	'size'   : 26}

	matplotlib.rc('font', **font)		
	plt.scatter(x,data2,s=area,c = np.random.rand(data2.shape[0]*data2.shape[1]))
	plt.xlabel('NodeID in NDN Network')
	plt.ylabel('Cache Size')
	plt.xlim([0,12])
	plt.ylim([-500,26500])
	plt.title('Projection of Observations on PrincipalComponents')
	plt.show()
	
def MyStem():
	m1 = np.loadtxt("Rcvd-Requests-AllNodes-PerSec-1.csv", delimiter=',')
	X1=np.matrix(m1)
	m2 = np.loadtxt("Rcvd-Requests-AllNodes-PerSec-2.csv", delimiter=',')
	X2=np.matrix(m2)
	m3 = np.loadtxt("Rcvd-Requests-AllNodes-PerSec-3.csv", delimiter=',')
	X3=np.matrix(m3)
	m4 = np.loadtxt("Rcvd-Requests-AllNodes-PerSec-4.csv", delimiter=',')
	X4=np.matrix(m4)
	m5 = np.loadtxt("Rcvd-Requests-AllNodes-PerSec-5.csv", delimiter=',')
	X5=np.matrix(m5)
	m6 = np.loadtxt("Rcvd-Requests-AllNodes-PerSec-6.csv", delimiter=',')
	X6=np.matrix(m6)
	m7 = np.loadtxt("Rcvd-Requests-AllNodes-PerSec-7.csv", delimiter=',')
	X7=np.matrix(m7)
	m8 = np.loadtxt("Rcvd-Requests-AllNodes-PerSec-8.csv", delimiter=',')
	X8=np.matrix(m8)
	m9 = np.loadtxt("Rcvd-Requests-AllNodes-PerSec-9.csv", delimiter=',')
	X9=np.matrix(m9)
	m10 = np.loadtxt("Rcvd-Requests-AllNodes-PerSec-10.csv", delimiter=',')
	X10=np.matrix(m10)
	m_mean = np.zeros((27,600))

	for i in range(m_mean.shape[0]):
		for j in range(m_mean.shape[1]):
			#print i
			m_mean.itemset((i,j),(X1.item(i,j)+X2.item(i,j)+X3.item(i,j)+X4.item(i,j)+X5.item(i,j)+X6.item(i,j)+X7.item(i,j)+X8.item(i,j)+X9.item(i,j)+X10.item(i,j))/10)
	
	X1_1stRow = m_mean[22,:].transpose()
	x_axis=np.linspace(1,600,600)
	
	font = {'family' : 'STIXGeneral',
        	'weight' : 'bold',
        	'size'   : 26}

	matplotlib.rc('font', **font)

	plt.scatter(x_axis,X1_1stRow)
	plt.plot(x_axis,X1_1stRow,label="Simulation")
	
	
	# 22
	a = (100/(x_axis**1.35+50))
	#plt.plot(x_axis,a,'^-r',label="y=100/(x+50)")
	plt.plot(x_axis,a,'^-r',label="Theory")
	plt.xlim([-0.2,600])
	plt.ylim([0,2])
	plt.xlabel('Time(sec)')
	plt.ylabel('Recieved_Requests')
	plt.legend(loc='upper right')
	#plt.ylabel('Traffic')
	#plt.title('Traffic_Preformance_of_Rtr6')
	plt.title('PendingInterestTable(PIT) of Core Router10')
	plt.show()

Degree = np.array([5,4,6,3,3,3,6,2,4,3,5])

Betwness = np.array([180,96,260,58,112,224,234,44,98,214,206])

Load     = np.array([238.9,149.9,310,1,165.9,270.1,287.1,131,2,236,258.1]) ###?????????

pit	 = np.array(PIT().transpose(),dtype='float32')

'''cs	 = np.array(CS().transpose(),dtype='float32')

Norm_final = np.zeros((11,3))
Norm_final2 = np.zeros((11,3))
Arr = np.zeros((11,4))
X_std = np.zeros((11,3))

sum_pit = np.sum(pit)
#pit_norm = np.zeros((11))

sum_cs = np.sum(cs)
#cs_norm = np.zeros((11))

sum_Betwness = np.sum(Betwness)
#Betwness_norm = np.zeros((11))

sum_Degree = np.sum(Degree)
#print Degree
#Degree_norm = np.zeros((11))

#print np.around(((Betwness*143000)/(sum_Betwness)))
#Norm_final2[:,0]= np.around(((Degree*143000)/(sum_Degree)))

print np.around(((Degree*143000)/(sum_Degree)))

Norm_final2[:,0]= np.around(((Betwness*143000)/(sum_Betwness)))
Norm_final2[:,1]= np.around(((pit)/(sum_pit))*143000)
Norm_final2[:,2]= np.around(((cs)/(sum_cs))*143000)
#print Norm_final2

##Without allocating weights##
#Norm_final[:,0]=((Degree))
Norm_final[:,0] = Betwness
Norm_final[:,1] = pit
Norm_final[:,2] = cs
#print Norm_final

Arr[:,0] = np.linspace(1,11,11)
Arr[:,1:4] = Norm_final[:,:]
##Sort the matrix according to Column 3 of the matrix##
Arr = Arr[np.argsort(Arr[:,3])]
#print Arr
Norm_final[:,:] = Arr[:,1:4]

min = 0
max = 0

for i in range(Norm_final2.shape[0]):
	for j in range(Norm_final2.shape[1]):
		if i == 0 and j == 0:
			min = 26000 - Norm_final2[i,j]
			#print min
		elif ((26000 - Norm_final2[i,j]) <= min and (26000 - Norm_final2[i,j]) >= 0):
			min = (26000 - Norm_final2[i,j])
			max = Norm_final2[i,j]
#print Norm_final2
#print min
#print max
for i in range(Norm_final2.shape[0]):
	for j in range(Norm_final2.shape[1]):
		if Norm_final2[i,j] >= 26000:
			Norm_final2[i,j] = max+100

X_std[:,0] = Norm_final2[:,0]/np.sum(Norm_final2[:,0])
X_std[:,1] = Norm_final2[:,1]/np.sum(Norm_final2[:,1])
X_std[:,2] = Norm_final2[:,2]/np.sum(Norm_final2[:,2])
#print X_std
X_std[:,0] = np.around((X_std[:,0]-np.min(X_std[:,0]))/(np.max(X_std[:,0])-np.min(X_std[:,0])),10)
X_std[:,1] = np.around((X_std[:,1]-np.min(X_std[:,1]))/(np.max(X_std[:,1])-np.min(X_std[:,1])),10)
X_std[:,2] = np.around((X_std[:,2]-np.min(X_std[:,2]))/(np.max(X_std[:,2])-np.min(X_std[:,2])),10)
#print X_std

data = np.zeros((33,1))
data[0:11,0] = X_std[:,0]
data[11:22,0] = X_std[:,1]
data[22:33,0] = X_std[:,2]
#MyPlot2DimReshaped(data)
#MyPlotNotReshaped(X_std)
My_KernelPCA(X_std)
#My_PCA(X_std)
X_std = StandardScaler().fit_transform(X_std)
min_X_std = np.min(X_std,axis=0)
X_std_new = (X_std+((-1)*min_X_std))
X_std_Norm = X_std_new/np.max(X_std_new)
O = X_std_Norm
MyImportance = np.zeros((11))

for i in range(O.shape[0]):
	for j in range(O.shape[1]):
		if O[i,j] >= 0 and O[i,j]< 0.1:
			MyImportance.itemset((i),MyImportance.item(i)+1)
		elif O[i,j] >= 0.1 and O[i,j]< 0.2:
			MyImportance.itemset((i),MyImportance.item(i)+2)
		elif O[i,j] >= 0.2 and O[i,j]< 0.3:
			MyImportance.itemset((i),MyImportance.item(i)+3)
		elif O[i,j] >= 0.3 and O[i,j]< 0.4:
			MyImportance.itemset((i),MyImportance.item(i)+4)
		elif O[i,j] >= 0.4 and O[i,j]< 0.5:
			MyImportance.itemset((i),MyImportance.item(i)+5)
		elif O[i,j] >= 0.5 and O[i,j]< 0.6:
			MyImportance.itemset((i),MyImportance.item(i)+6)
		elif O[i,j] >= 0.6 and O[i,j]< 0.7:
			MyImportance.itemset((i),MyImportance.item(i)+7)
		elif O[i,j] >= 0.7 and O[i,j]< 0.8:
			MyImportance.itemset((i),MyImportance.item(i)+8)
		elif O[i,j] >= 0.8 and O[i,j]< 0.9:
			MyImportance.itemset((i),MyImportance.item(i)+9)
		elif O[i,j] >= 0.9 and O[i,j]<= 1:
			MyImportance.itemset((i),MyImportance.item(i)+10)
print MyImportance
MyImportance_Weight = MyImportance/np.sum(MyImportance)
#print MyImportance_Weight
MyImportance_CS = np.around(MyImportance_Weight*143000,0)
print np.transpose(MyImportance_CS)
#MyPCAPlot(MyImportance_CS)'''
