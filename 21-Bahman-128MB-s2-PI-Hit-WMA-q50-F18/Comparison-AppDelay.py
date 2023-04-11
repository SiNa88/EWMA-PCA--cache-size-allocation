
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pylab import *
from matplotlib.font_manager import FontProperties, findfont

def ExpMovingAvg(values, window):
	weights = np.exp(np.linspace(-1.,0.,window))
	weights /= weights.sum()
	a = np.convolve(values,weights,mode='full') [:len(values)]
	return a
def MovingAvg(values, window):

	a = np.convolve(values,np.ones((window,))/window,mode='full') [:len(values)]
	return a

def CS():
	m1_HomoCons = np.loadtxt("Homo/Homo-AppDelay-PerSec-1.csv", delimiter=',')
	X1_HomoCons=np.matrix(m1_HomoCons)
	'''m2_HomoCons = np.loadtxt("Homo/Homo-AppDelay-PerSec-2.csv", delimiter=',')
	X2_HomoCons=np.matrix(m2_HomoCons)
	m3_HomoCons = np.loadtxt("Homo/Homo-AppDelay-PerSec-3.csv", delimiter=',')
	X3_HomoCons=np.matrix(m3_HomoCons)
	m4_HomoCons = np.loadtxt("Homo/Homo-AppDelay-PerSec-4.csv", delimiter=',')
	X4_HomoCons=np.matrix(m4_HomoCons)
	m5_HomoCons = np.loadtxt("Homo/Homo-AppDelay-PerSec-5.csv", delimiter=',')
	X5_HomoCons=np.matrix(m5_HomoCons)
	m6_HomoCons = np.loadtxt("Homo/Homo-AppDelay-PerSec-6.csv", delimiter=',')
	X6_HomoCons=np.matrix(m6_HomoCons)
	m7_HomoCons = np.loadtxt("Homo/Homo-AppDelay-PerSec-7.csv", delimiter=',')
	X7_HomoCons=np.matrix(m7_HomoCons)
	m8_HomoCons = np.loadtxt("Homo/Homo-AppDelay-PerSec-8.csv", delimiter=',')
	X8_HomoCons=np.matrix(m8_HomoCons)
	m9_HomoCons = np.loadtxt("Homo/Homo-AppDelay-PerSec-9.csv", delimiter=',')
	X9_HomoCons=np.matrix(m9_HomoCons)
	m10_HomoCons = np.loadtxt("Homo/Homo-AppDelay-PerSec-10.csv", delimiter=',')
	X10_HomoCons=np.matrix(m1_HomoCons)'''
	
	m_mean_HomoCons = np.zeros((12,600))
	
	m1_PCACons = np.loadtxt("PCA-With Betweenness/PCA-AppDelay-PerSec-1.csv", delimiter=',')
	X1_PCACons=np.matrix(m1_PCACons)
	'''m2_PCACons = np.loadtxt("PCA-With Betweenness/PCA-AppDelay-PerSec-2.csv", delimiter=',')
	X2_PCACons=np.matrix(m2_PCACons)
	m3_PCACons = np.loadtxt("PCA-With Betweenness/PCA-AppDelay-PerSec-3.csv", delimiter=',')
	X3_PCACons=np.matrix(m3_PCACons)
	m4_PCACons = np.loadtxt("PCA-With Betweenness/PCA-AppDelay-PerSec-4.csv", delimiter=',')
	X4_PCACons=np.matrix(m4_PCACons)
	m5_PCACons = np.loadtxt("PCA-With Betweenness/PCA-AppDelay-PerSec-5.csv", delimiter=',')
	X5_PCACons=np.matrix(m5_PCACons)
	m6_PCACons = np.loadtxt("PCA-With Betweenness/PCA-AppDelay-PerSec-6.csv", delimiter=',')
	X6_PCACons=np.matrix(m6_PCACons)
	m7_PCACons = np.loadtxt("PCA-With Betweenness/PCA-AppDelay-PerSec-7.csv", delimiter=',')
	X7_PCACons=np.matrix(m7_PCACons)
	m8_PCACons = np.loadtxt("PCA-With Betweenness/PCA-AppDelay-PerSec-8.csv", delimiter=',')
	X8_PCACons=np.matrix(m8_PCACons)
	m9_PCACons = np.loadtxt("PCA-With Betweenness/PCA-AppDelay-PerSec-9.csv", delimiter=',')
	X9_PCACons=np.matrix(m9_PCACons)
	m10_PCACons = np.loadtxt("PCA-With Betweenness/PCA-AppDelay-PerSec-10.csv", delimiter=',')
	X10_PCACons=np.matrix(m10_PCACons)'''
	
	m_mean_PCACons = np.zeros((12,600))

	m1_DegreeCons = np.loadtxt("Deg/Deg-AppDelay-PerSec-1.csv", delimiter=',')
	X1_DegreeCons=np.matrix(m1_DegreeCons)
	'''m2_DegreeCons = np.loadtxt("Deg/Deg-AppDelay-PerSec-2.csv", delimiter=',')
	X2_DegreeCons=np.matrix(m2_DegreeCons)
	m3_DegreeCons = np.loadtxt("Deg/Deg-AppDelay-PerSec-3.csv", delimiter=',')
	X3_DegreeCons=np.matrix(m3_DegreeCons)
	m4_DegreeCons = np.loadtxt("Deg/Deg-AppDelay-PerSec-4.csv", delimiter=',')
	X4_DegreeCons=np.matrix(m4_DegreeCons)
	m5_DegreeCons = np.loadtxt("Deg/Deg-AppDelay-PerSec-5.csv", delimiter=',')
	X5_DegreeCons=np.matrix(m5_DegreeCons)
	m6_DegreeCons = np.loadtxt("Deg/Deg-AppDelay-PerSec-6.csv", delimiter=',')
	X6_DegreeCons=np.matrix(m6_DegreeCons)
	m7_DegreeCons = np.loadtxt("Deg/Deg-AppDelay-PerSec-7.csv", delimiter=',')
	X7_DegreeCons=np.matrix(m7_DegreeCons)
	m8_DegreeCons = np.loadtxt("Deg/Deg-AppDelay-PerSec-8.csv", delimiter=',')
	X8_DegreeCons=np.matrix(m8_DegreeCons)
	m9_DegreeCons = np.loadtxt("Deg/Deg-AppDelay-PerSec-9.csv", delimiter=',')
	X9_DegreeCons=np.matrix(m9_DegreeCons)
	m10_DegreeCons = np.loadtxt("Deg/Deg-AppDelay-PerSec-10.csv", delimiter=',')
	X10_DegreeCons=np.matrix(m10_DegreeCons)'''
	
	m_mean_DegreeCons = np.zeros((12,600))	
	
	for i in range(m_mean_PCACons.shape[0]):
		for j in range(m_mean_PCACons.shape[1]):
			m_mean_HomoCons.itemset((i,j),(X1_HomoCons.item(i,j)))
			#+X2_HomoCons.item(i,j)+X3_HomoCons.item(i,j)+X4_HomoCons.item(i,j)+X5_HomoCons.item(i,j)+X6_HomoCons.item(i,j)+X7_HomoCons.item(i,j)+X8_HomoCons.item(i,j)+X9_HomoCons.item(i,j)+X10_HomoCons.item(i,j))/10)
			m_mean_PCACons.itemset((i,j),(X1_PCACons.item(i,j)))
			#+X2_PCACons.item(i,j)+X3_PCACons.item(i,j)+X4_PCACons.item(i,j)+X5_PCACons.item(i,j)+X6_PCACons.item(i,j)+X7_PCACons.item(i,j)+X8_PCACons.item(i,j)+X9_PCACons.item(i,j)+X10_PCACons.item(i,j))/10)
			m_mean_DegreeCons.itemset((i,j),(X1_DegreeCons.item(i,j)))
			#+X2_DegreeCons.item(i,j)+X3_DegreeCons.item(i,j)+X4_DegreeCons.item(i,j)+X5_DegreeCons.item(i,j)+X6_DegreeCons.item(i,j)+X7_DegreeCons.item(i,j)+X8_DegreeCons.item(i,j)+X9_DegreeCons.item(i,j)+X10_DegreeCons.item(i,j))/10)

	font = {'family' : 'normal','weight' : 'bold','size'   : 23}
	
	matplotlib.rc('font', **font)
	tedaad = 6
	x = np.linspace(0,tedaad,tedaad)
	#AVG_NoCache_Cons = np.zeros((1,600))
	#AVG_NoCache_Cons = np.copy(np.sum(ConsRateNoCache, axis=0)/12)
	AVG_Homo_Cons = np.zeros((1,600))
	AVG_Homo_Cons = np.copy(np.sum(m_mean_HomoCons[0:12,0:600], axis=0)/12)
	AVG_Degree_Cons = np.zeros((1,600))
	AVG_Degree_Cons = np.copy(np.sum(m_mean_DegreeCons[0:12,0:600], axis=0)/12)
	AVG_PCA_Cons = np.zeros((1,600))
	AVG_PCA_Cons = np.copy(np.sum(m_mean_PCACons[0:12,0:600], axis=0)/12)
	counter = 0
	
	num_of_chunks = 0

	for i in range(AVG_PCA_Cons.shape[0]):
		if AVG_PCA_Cons[i] != 0 and AVG_Degree_Cons[i] != 0 and AVG_Homo_Cons[i] != 0:
			num_of_chunks = num_of_chunks + 1
		if AVG_PCA_Cons[i]< AVG_Degree_Cons[i] or AVG_PCA_Cons[i] == AVG_Degree_Cons[i] or AVG_PCA_Cons[i]< AVG_Homo_Cons[i] or AVG_PCA_Cons[i] == AVG_Homo_Cons[i]:
			counter = counter + 1	
	
	#print  np.around(np.average(AVG_Homo_Cons)),np.around(np.average(AVG_Degree_Cons)),np.around(np.average(AVG_PCA_Cons))
	
	print  np.around(np.average(m_mean_HomoCons)/120000,6),"\n",np.around(np.average(m_mean_DegreeCons)/120000,6),"\n",np.around(np.average(m_mean_PCACons)/120000,6),"\n"
	#######print counter
	print num_of_chunks
	#print np.around(np.average(m_mean_HomoCons[0,:]+m_mean_HomoCons[1,:]+m_mean_HomoCons[2,:]+m_mean_HomoCons[3,:]+m_mean_HomoCons[4,:]+m_mean_HomoCons[5,:]+m_mean_HomoCons[6,:]+m_mean_HomoCons[7,:]+m_mean_HomoCons[8,:]+m_mean_HomoCons[9,:]+m_mean_HomoCons[10,:]+m_mean_HomoCons[11,:]),6),np.around(np.average(m_mean_DegreeCons[0,:]+m_mean_DegreeCons[1,:]+m_mean_DegreeCons[2,:]+m_mean_DegreeCons[3,:]+m_mean_DegreeCons[4,:]+m_mean_DegreeCons[5,:]+m_mean_DegreeCons[6,:]+m_mean_DegreeCons[7,:]+m_mean_DegreeCons[8,:]+m_mean_DegreeCons[9,:]+m_mean_DegreeCons[10,:]+m_mean_DegreeCons[11,:]),6),np.around(np.average(m_mean_PCACons[0,:]+m_mean_PCACons[1,:]+m_mean_PCACons[2,:]+m_mean_PCACons[3,:]+m_mean_PCACons[4,:]+m_mean_PCACons[5,:]+m_mean_PCACons[6,:]+m_mean_PCACons[7,:]+m_mean_PCACons[8,:]+m_mean_PCACons[9,:]+m_mean_PCACons[10,:]+m_mean_PCACons[11,:]),6)

	Smooth_Homo_Cons = np.zeros((tedaad,1))#np.copy(np.zeros((1,tedaad)))
	Smooth_Degree_Cons = np.zeros((tedaad,1))#np.copy(np.zeros((1,tedaad)))
	Smooth_PCA_Cons = np.zeros((tedaad,1))#np.copy(np.zeros((1,tedaad)))
	j = 0
	for i in range(AVG_PCA_Cons.shape[0]):
		if ((j+1)*(600/tedaad)) <= i:
			j = j + 1
		Smooth_Homo_Cons[j] = Smooth_Homo_Cons[j] + AVG_Homo_Cons[i]
		Smooth_Degree_Cons[j] = Smooth_Degree_Cons[j] + AVG_Degree_Cons[i]
		Smooth_PCA_Cons[j] = Smooth_PCA_Cons[j] + AVG_PCA_Cons[i]
		#print i,j
	for i in range(Smooth_PCA_Cons.shape[0]):
		Smooth_Homo_Cons [i] = Smooth_Homo_Cons [i] / (600/tedaad)
		Smooth_Degree_Cons [i] = Smooth_Degree_Cons [i] / (600/tedaad)
		Smooth_PCA_Cons [i] = Smooth_PCA_Cons [i] / (600/tedaad)
	#Smooth_Homo_Cons = np.copy(Smooth_Homo_Cons)
	#Smooth_Degree_Cons = np.copy(Smooth_Degree_Cons)
	#Smooth_PCA_Cons = np.copy(Smooth_PCA_Cons)

	##print np.around(Smooth_Homo_Cons)
	##print np.around(Smooth_Degree_Cons)
	##print np.around(Smooth_PCA_Cons)
	plt.plot(x,np.around(Smooth_Homo_Cons/120000,6),'.-c',label="Uniform")
	plt.plot(x,np.around(Smooth_Degree_Cons/120000,6),'o-b',label="Degree")
	plt.plot(x,np.around(Smooth_PCA_Cons/120000,6),'^-r',label="ProposedMethod")
	plt.legend(loc='upper right')
	plt.title('Comparison in Consumer Routers Interest-Data Delay')
	plt.xlim([0,tedaad])	
	plt.ylim([0.6,1.4])
	plt.xlabel('Simulaton Time(*100 Seconds)')
	plt.ylabel('Delay (MilliSecond)')
	plt.show()
np.array(CS(),dtype='float32')
#fig = plt.figure()
#plt.plot(data)
#fig.suptitle('test title')
#plt.xlabel('xlabel')
#plt.ylabel('ylabel')
#fig.savefig('test.jpg')
