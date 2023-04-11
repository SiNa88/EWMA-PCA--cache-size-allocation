
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pylab import *

def ExpMovingAvg(values, window):
	weights = np.exp(np.linspace(-1.,0.,window))
	weights /= weights.sum()
	a = np.convolve(values,weights,mode='full') [:len(values)]
	return a
def MovingAvg(values, window):

	a = np.convolve(values,np.ones((window,))/window,mode='full') [:len(values)]
	return a
def CS():
	
	
	m1_HomoHit = np.loadtxt("Homo/Homo-Hit-Src-PerSec-1.csv", delimiter=',')
	X1_HomoHit=np.matrix(m1_HomoHit)
	'''m2_HomoHit = np.loadtxt("Homo/Homo-Hit-Src-PerSec-2.csv", delimiter=',')
	X2_HomoHit=np.matrix(m2_HomoHit)
	m3_HomoHit = np.loadtxt("Homo/Homo-Hit-Src-PerSec-3.csv", delimiter=',')
	X3_HomoHit=np.matrix(m3_HomoHit)
	m4_HomoHit = np.loadtxt("Homo/Homo-Hit-Src-PerSec-4.csv", delimiter=',')
	X4_HomoHit=np.matrix(m4_HomoHit)
	m5_HomoHit = np.loadtxt("Homo/Homo-Hit-Src-PerSec-5.csv", delimiter=',')
	X5_HomoHit=np.matrix(m5_HomoHit)
	m6_HomoHit = np.loadtxt("Homo/Homo-Hit-Src-PerSec-6.csv", delimiter=',')
	X6_HomoHit=np.matrix(m6_HomoHit)
	m7_HomoHit = np.loadtxt("Homo/Homo-Hit-Src-PerSec-7.csv", delimiter=',')
	X7_HomoHit=np.matrix(m7_HomoHit)
	m8_HomoHit = np.loadtxt("Homo/Homo-Hit-Src-PerSec-8.csv", delimiter=',')
	X8_HomoHit=np.matrix(m8_HomoHit)
	m9_HomoHit = np.loadtxt("Homo/Homo-Hit-Src-PerSec-9.csv", delimiter=',')
	X9_HomoHit=np.matrix(m9_HomoHit)
	m10_HomoHit = np.loadtxt("Homo/Homo-Hit-Src-PerSec-10.csv", delimiter=',')
	X10_HomoHit=np.matrix(m10_HomoHit)'''
	m_mean_HomoHit = np.zeros((4,600))	
	
	
	m1_HomoHitMiss = np.loadtxt("Homo/Homo-HitMiss-Src-PerSec-1.csv", delimiter=',')
	X1_HomoHitMiss=np.matrix(m1_HomoHitMiss)
	'''m2_HomoHitMiss = np.loadtxt("Homo/Homo-HitMiss-Src-PerSec-2.csv", delimiter=',')
	X2_HomoHitMiss=np.matrix(m2_HomoHitMiss)
	m3_HomoHitMiss = np.loadtxt("Homo/Homo-HitMiss-Src-PerSec-3.csv", delimiter=',')
	X3_HomoHitMiss=np.matrix(m3_HomoHitMiss)
	m4_HomoHitMiss = np.loadtxt("Homo/Homo-HitMiss-Src-PerSec-4.csv", delimiter=',')
	X4_HomoHitMiss=np.matrix(m4_HomoHitMiss)
	m5_HomoHitMiss = np.loadtxt("Homo/Homo-HitMiss-Src-PerSec-5.csv", delimiter=',')
	X5_HomoHitMiss=np.matrix(m5_HomoHitMiss)
	m6_HomoHitMiss = np.loadtxt("Homo/Homo-HitMiss-Src-PerSec-6.csv", delimiter=',')
	X6_HomoHitMiss=np.matrix(m6_HomoHitMiss)
	m7_HomoHitMiss = np.loadtxt("Homo/Homo-HitMiss-Src-PerSec-7.csv", delimiter=',')
	X7_HomoHitMiss=np.matrix(m7_HomoHitMiss)
	m8_HomoHitMiss = np.loadtxt("Homo/Homo-HitMiss-Src-PerSec-8.csv", delimiter=',')
	X8_HomoHitMiss=np.matrix(m8_HomoHitMiss)
	m9_HomoHitMiss = np.loadtxt("Homo/Homo-HitMiss-Src-PerSec-9.csv", delimiter=',')
	X9_HomoHitMiss=np.matrix(m9_HomoHitMiss)
	m10_HomoHitMiss = np.loadtxt("Homo/Homo-HitMiss-Src-PerSec-10.csv", delimiter=',')
	X10_HomoHitMiss=np.matrix(m10_HomoHitMiss)'''
	
	m_mean_HomoHitMiss = np.zeros((4,600))
	HitRateHomo=np.zeros((4,600))

	
	m1_PCAHit = np.loadtxt("PCA-With Betweenness/PCA-Hit-Src-PerSec-1.csv", delimiter=',')
	X1_PCAHit=np.matrix(m1_PCAHit)
	'''m2_PCAHit = np.loadtxt("PCA-With Betweenness/PCA-Hit-Src-PerSec-2.csv", delimiter=',')
	X2_PCAHit=np.matrix(m2_PCAHit)
	m3_PCAHit = np.loadtxt("PCA-With Betweenness/PCA-Hit-Src-PerSec-3.csv", delimiter=',')
	X3_PCAHit=np.matrix(m3_PCAHit)
	m4_PCAHit = np.loadtxt("PCA-With Betweenness/PCA-Hit-Src-PerSec-4.csv", delimiter=',')
	X4_PCAHit=np.matrix(m4_PCAHit)
	m5_PCAHit = np.loadtxt("PCA-With Betweenness/PCA-Hit-Src-PerSec-5.csv", delimiter=',')
	X5_PCAHit=np.matrix(m5_PCAHit)
	m6_PCAHit = np.loadtxt("PCA-With Betweenness/PCA-Hit-Src-PerSec-6.csv", delimiter=',')
	X6_PCAHit=np.matrix(m6_PCAHit)
	m7_PCAHit = np.loadtxt("PCA-With Betweenness/PCA-Hit-Src-PerSec-7.csv", delimiter=',')
	X7_PCAHit=np.matrix(m7_PCAHit)
	m8_PCAHit = np.loadtxt("PCA-With Betweenness/PCA-Hit-Src-PerSec-8.csv", delimiter=',')
	X8_PCAHit=np.matrix(m8_PCAHit)
	m9_PCAHit = np.loadtxt("PCA-With Betweenness/PCA-Hit-Src-PerSec-9.csv", delimiter=',')
	X9_PCAHit=np.matrix(m9_PCAHit)
	m10_PCAHit = np.loadtxt("PCA-With Betweenness/PCA-Hit-Src-PerSec-10.csv", delimiter=',')
	X10_PCAHit=np.matrix(m10_PCAHit)'''
	m_mean_PCAHit = np.zeros((4,600))	
	
	
	m1_PCAHitMiss = np.loadtxt("PCA-With Betweenness/PCA-HitMiss-Src-PerSec-1.csv", delimiter=',')
	X1_PCAHitMiss=np.matrix(m1_PCAHitMiss)
	'''m2_PCAHitMiss = np.loadtxt("PCA-With Betweenness/PCA-HitMiss-Src-PerSec-2.csv", delimiter=',')
	X2_PCAHitMiss=np.matrix(m2_PCAHitMiss)
	m3_PCAHitMiss = np.loadtxt("PCA-With Betweenness/PCA-HitMiss-Src-PerSec-3.csv", delimiter=',')
	X3_PCAHitMiss=np.matrix(m3_PCAHitMiss)
	m4_PCAHitMiss = np.loadtxt("PCA-With Betweenness/PCA-HitMiss-Src-PerSec-4.csv", delimiter=',')
	X4_PCAHitMiss=np.matrix(m4_PCAHitMiss)
	m5_PCAHitMiss = np.loadtxt("PCA-With Betweenness/PCA-HitMiss-Src-PerSec-5.csv", delimiter=',')
	X5_PCAHitMiss=np.matrix(m5_PCAHitMiss)
	m6_PCAHitMiss = np.loadtxt("PCA-With Betweenness/PCA-HitMiss-Src-PerSec-6.csv", delimiter=',')
	X6_PCAHitMiss=np.matrix(m6_PCAHitMiss)
	m7_PCAHitMiss = np.loadtxt("PCA-With Betweenness/PCA-HitMiss-Src-PerSec-7.csv", delimiter=',')
	X7_PCAHitMiss=np.matrix(m7_PCAHitMiss)
	m8_PCAHitMiss = np.loadtxt("PCA-With Betweenness/PCA-HitMiss-Src-PerSec-8.csv", delimiter=',')
	X8_PCAHitMiss=np.matrix(m8_PCAHitMiss)
	m9_PCAHitMiss = np.loadtxt("PCA-With Betweenness/PCA-HitMiss-Src-PerSec-9.csv", delimiter=',')
	X9_PCAHitMiss=np.matrix(m9_PCAHitMiss)
	m10_PCAHitMiss = np.loadtxt("PCA-With Betweenness/PCA-HitMiss-Src-PerSec-10.csv", delimiter=',')
	X10_PCAHitMiss=np.matrix(m10_PCAHitMiss)'''
	
	m_mean_PCAHitMiss = np.zeros((4,600))
	HitRatePCA=np.zeros((4,600))
	
	
	m1_DegreeHit = np.loadtxt("Deg/Deg-Hit-Src-PerSec-1.csv", delimiter=',')
	X1_DegreeHit=np.matrix(m1_DegreeHit)
	'''m2_DegreeHit = np.loadtxt("Deg/Deg-Hit-Src-PerSec-2.csv", delimiter=',')
	X2_DegreeHit=np.matrix(m2_DegreeHit)
	m3_DegreeHit = np.loadtxt("Deg/Deg-Hit-Src-PerSec-3.csv", delimiter=',')
	X3_DegreeHit=np.matrix(m3_DegreeHit)
	m4_DegreeHit = np.loadtxt("Deg/Deg-Hit-Src-PerSec-4.csv", delimiter=',')
	X4_DegreeHit=np.matrix(m4_DegreeHit)
	m5_DegreeHit = np.loadtxt("Deg/Deg-Hit-Src-PerSec-5.csv", delimiter=',')
	X5_DegreeHit=np.matrix(m5_DegreeHit)
	m6_DegreeHit = np.loadtxt("Deg/Deg-Hit-Src-PerSec-6.csv", delimiter=',')
	X6_DegreeHit=np.matrix(m6_DegreeHit)
	m7_DegreeHit = np.loadtxt("Deg/Deg-Hit-Src-PerSec-7.csv", delimiter=',')
	X7_DegreeHit=np.matrix(m7_DegreeHit)
	m8_DegreeHit = np.loadtxt("Deg/Deg-Hit-Src-PerSec-8.csv", delimiter=',')
	X8_DegreeHit=np.matrix(m8_DegreeHit)
	m9_DegreeHit = np.loadtxt("Deg/Deg-Hit-Src-PerSec-9.csv", delimiter=',')
	X9_DegreeHit=np.matrix(m9_DegreeHit)
	m10_DegreeHit = np.loadtxt("Deg/Deg-Hit-Src-PerSec-10.csv", delimiter=',')
	X10_DegreeHit=np.matrix(m10_DegreeHit)'''
	
	m_mean_DegreeHit = np.zeros((4,600))	
	
	m1_DegreeHitMiss = np.loadtxt("Deg/Deg-HitMiss-Src-PerSec-1.csv", delimiter=',')
	X1_DegreeHitMiss=np.matrix(m1_DegreeHitMiss)
	'''m2_DegreeHitMiss = np.loadtxt("Deg/Deg-HitMiss-Src-PerSec-2.csv", delimiter=',')
	X2_DegreeHitMiss=np.matrix(m2_DegreeHitMiss)
	m3_DegreeHitMiss = np.loadtxt("Deg/Deg-HitMiss-Src-PerSec-3.csv", delimiter=',')
	X3_DegreeHitMiss=np.matrix(m3_DegreeHitMiss)
	m4_DegreeHitMiss = np.loadtxt("Deg/Deg-HitMiss-Src-PerSec-4.csv", delimiter=',')
	X4_DegreeHitMiss=np.matrix(m4_DegreeHitMiss)
	m5_DegreeHitMiss = np.loadtxt("Deg/Deg-HitMiss-Src-PerSec-5.csv", delimiter=',')
	X5_DegreeHitMiss=np.matrix(m5_DegreeHitMiss)
	m6_DegreeHitMiss = np.loadtxt("Deg/Deg-HitMiss-Src-PerSec-6.csv", delimiter=',')
	X6_DegreeHitMiss=np.matrix(m6_DegreeHitMiss)
	m7_DegreeHitMiss = np.loadtxt("Deg/Deg-HitMiss-Src-PerSec-7.csv", delimiter=',')
	X7_DegreeHitMiss=np.matrix(m7_DegreeHitMiss)
	m8_DegreeHitMiss = np.loadtxt("Deg/Deg-HitMiss-Src-PerSec-8.csv", delimiter=',')
	X8_DegreeHitMiss=np.matrix(m8_DegreeHitMiss)
	m9_DegreeHitMiss = np.loadtxt("Deg/Deg-HitMiss-Src-PerSec-9.csv", delimiter=',')
	X9_DegreeHitMiss=np.matrix(m9_DegreeHitMiss)
	m10_DegreeHitMiss = np.loadtxt("Deg/Deg-HitMiss-Src-PerSec-10.csv", delimiter=',')
	X10_DegreeHitMiss=np.matrix(m10_DegreeHitMiss)'''

	m_mean_DegreeHitMiss = np.zeros((4,600))
	HitRateDegree=np.zeros((4,600))
	
	for i in range(m_mean_PCAHitMiss.shape[0]):
		for j in range(m_mean_PCAHitMiss.shape[1]):
			
			m_mean_HomoHit.itemset((i,j),(X1_HomoHit.item(i,j)))
			#+X2_HomoHit.item(i,j)+X3_HomoHit.item(i,j)+X4_HomoHit.item(i,j)+X5_HomoHit.item(i,j)+X6_HomoHit.item(i,j)+X7_HomoHit.item(i,j)+X8_HomoHit.item(i,j)+X9_HomoHit.item(i,j)+X10_HomoHit.item(i,j))/10)
			m_mean_HomoHitMiss.itemset((i,j),(X1_HomoHitMiss.item(i,j)))
			#+X2_HomoHitMiss.item(i,j)+X3_HomoHitMiss.item(i,j)+X4_HomoHitMiss.item(i,j)+X5_HomoHitMiss.item(i,j)+X6_HomoHitMiss.item(i,j)+X7_HomoHitMiss.item(i,j)+X8_HomoHitMiss.item(i,j)+X9_HomoHitMiss.item(i,j)+X10_HomoHitMiss.item(i,j))/10)
			#std_HomoHitMiss.itemset((i,j),np.std([X1_HomoHit.item(i,j),X2_HomoHit.item(i,j),X3_HomoHit.item(i,j),X4_HomoHit.item(i,j),X5_HomoHit.item(i,j),X6_HomoHit.item(i,j),X7_HomoHit.item(i,j),X8_HomoHit.item(i,j),X9_HomoHit.item(i,j),X10_HomoHit.item(i,j)]))
			if m_mean_HomoHitMiss.item(i,j) != 0 :
				HitRateHomo.itemset((i,j),m_mean_HomoHit.item(i,j)/m_mean_HomoHitMiss.item(i,j))
			else:
				HitRateHomo.itemset((i,j),0)

			m_mean_PCAHit.itemset((i,j),(X1_PCAHit.item(i,j)))
			#+X2_PCAHit.item(i,j)+X3_PCAHit.item(i,j)+X4_PCAHit.item(i,j)+X5_PCAHit.item(i,j)+X6_PCAHit.item(i,j)+X7_PCAHit.item(i,j)+X8_PCAHit.item(i,j)+X9_PCAHit.item(i,j)+X10_PCAHit.item(i,j))/10)
			m_mean_PCAHitMiss.itemset((i,j),(X1_PCAHitMiss.item(i,j)))
			#+X2_PCAHitMiss.item(i,j)+X3_PCAHitMiss.item(i,j)+X4_PCAHitMiss.item(i,j)+X5_PCAHitMiss.item(i,j)+X6_PCAHitMiss.item(i,j)+X7_PCAHitMiss.item(i,j)+X8_PCAHitMiss.item(i,j)+X9_PCAHitMiss.item(i,j)+X10_PCAHitMiss.item(i,j))/10)
			if m_mean_PCAHitMiss.item(i,j) != 0 :
				HitRatePCA.itemset((i,j),m_mean_PCAHit.item(i,j)/m_mean_PCAHitMiss.item(i,j))
			else:
				HitRatePCA.itemset((i,j),0)	
			m_mean_DegreeHit.itemset((i,j),(X1_DegreeHit.item(i,j)))
			#+X2_DegreeHit.item(i,j)+X3_DegreeHit.item(i,j)+X4_DegreeHit.item(i,j)+X5_DegreeHit.item(i,j)+X6_DegreeHit.item(i,j)+X7_DegreeHit.item(i,j)+X8_DegreeHit.item(i,j)+X9_DegreeHit.item(i,j)+X10_DegreeHit.item(i,j))/10)

			m_mean_DegreeHitMiss.itemset((i,j),(X1_DegreeHitMiss.item(i,j)))
			#+X2_DegreeHitMiss.item(i,j)+X3_DegreeHitMiss.item(i,j)+X4_DegreeHitMiss.item(i,j)+X5_DegreeHitMiss.item(i,j)+X6_DegreeHitMiss.item(i,j)+X7_DegreeHitMiss.item(i,j)+X8_DegreeHitMiss.item(i,j)+X9_DegreeHitMiss.item(i,j)+X10_DegreeHitMiss.item(i,j))/10)
			if m_mean_DegreeHitMiss.item(i,j) != 0 :
				HitRateDegree.itemset((i,j),m_mean_DegreeHit.item(i,j)/m_mean_DegreeHitMiss.item(i,j))
			else:
				HitRateDegree.itemset((i,j),0)
			

	font = {'family' : 'normal','weight' : 'bold',        	'size'   : 24}

	matplotlib.rc('font', **font)
	tedaad = 6
	x = np.linspace(0,tedaad,tedaad)
	#AVG_NoCache = np.zeros((1,600))
	#AVG_NoCache = np.sum(HitRateNoCache, axis=0)/4
	AVG_Homo = np.zeros((1,600))
	AVG_Homo = np.sum(HitRateHomo, axis=0)/4
	AVG_Degree = np.zeros((1,600))
	AVG_Degree = np.sum(HitRateDegree, axis=0)/4
	AVG_PCA = np.zeros((1,600))
	AVG_PCA = np.sum(HitRatePCA, axis=0)/4	
	counter = 0
	for i in range(AVG_PCA.shape[0]):
		if AVG_PCA[i] < AVG_Degree[i] or AVG_PCA[i] == AVG_Degree[i] or AVG_PCA[i]< AVG_Homo[i] or AVG_PCA[i] == AVG_Homo[i]:
			#print "[",i,"\t"#",",AVG_Degree[i],",",AVG_PCA[i],"],"
			counter = counter + 1

	print np.around(np.average(np.copy(AVG_Homo)),6),"\n",np.around(np.average(np.copy(AVG_Degree)),6),"\n",np.around(np.average(np.copy(AVG_PCA)),6),"\n"
	
	#print np.around(np.average(np.copy(np.sum(HitRateHomo, axis=0)/4)),6),np.around(np.average(np.copy(np.sum(HitRateDegree, axis=0)/4)),6),np.around(np.average(np.copy(np.sum(HitRatePCA, axis=0)/4)),6)
	#####print counter
	#print np.around(np.average(HitRateHomo[0,:]+HitRateHomo[1,:]+HitRateHomo[2,:]+HitRateHomo[3,:]),6),np.around(np.average(HitRateDegree[0,:]+HitRateDegree[1,:]+HitRateDegree[2,:]+HitRateDegree[3,:]),6),np.around(np.average(HitRatePCA[0,:]+HitRatePCA[1,:]+HitRatePCA[2,:]+HitRatePCA[3,:]),6)
	Smooth_Homo = np.zeros((tedaad,1))#np.copy(np.zeros((1,tedaad)))
	Smooth_Degree = np.zeros((tedaad,1))#np.copy(np.zeros((1,tedaad)))
	Smooth_PCA = np.zeros((tedaad,1))#np.copy(np.zeros((1,tedaad)))
	j = 0
	for i in range(AVG_PCA.shape[0]):
		if ((j+1)*(600/tedaad)) <= i:
			j = j + 1
		Smooth_Homo[j] = Smooth_Homo[j] + AVG_Homo[i]
		Smooth_Degree[j] = Smooth_Degree[j] + AVG_Degree[i]
		Smooth_PCA[j] = Smooth_PCA[j] + AVG_PCA[i]
		#print i,j
	for i in range(Smooth_PCA.shape[0]):
		Smooth_Homo [i] = Smooth_Homo [i] / (600/tedaad)
		Smooth_Degree [i] = Smooth_Degree [i] / (600/tedaad)
		Smooth_PCA [i] = Smooth_PCA [i] / (600/tedaad)
	#Smooth_Homo = np.copy(Smooth_Homo)
	#Smooth_Degree = np.copy(Smooth_Degree)
	#Smooth_PCA = np.copy(Smooth_PCA)
	###print Smooth_Homo
	###print Smooth_Degree
	###print Smooth_PCA
	plt.plot(x,Smooth_Homo,'.-g',label="Uniform")
	plt.plot(x,Smooth_Degree,'o-b',label="Degree")
	plt.plot(x,Smooth_PCA,'^-r',label="ProposedMethod")
	plt.legend(loc='upper left')
	#plt.title('Comparison in Source ContentStore Hit (q=2.5, s=2.0)')
	plt.title('Comparison in Source ContentStore Hit')	
	plt.xlim([0,tedaad])
	plt.ylim([0.00,0.02])
	plt.xlabel('SimulationTime(*100 Seconds)')
	plt.ylabel('HitRatio')
	plt.show()
np.array(CS(),dtype='float32')
#fig = plt.figure()
#plt.plot(data)
#fig.suptitle('test title')
#plt.xlabel('xlabel')
#plt.ylabel('ylabel')
#fig.savefig('test.jpg')
