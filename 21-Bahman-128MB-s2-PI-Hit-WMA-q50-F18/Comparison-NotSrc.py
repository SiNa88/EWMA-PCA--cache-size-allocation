
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
	m1_HomoHit = np.loadtxt("Homo/Homo-Hit-NotSrc-PerSec-1.csv", delimiter=',')
	X1_HomoHit=np.matrix(m1_HomoHit)
	'''m2_HomoHit = np.loadtxt("Homo/Homo-Hit-NotSrc-PerSec-2.csv", delimiter=',')
	X2_HomoHit=np.matrix(m2_HomoHit)
	m3_HomoHit = np.loadtxt("Homo/Homo-Hit-NotSrc-PerSec-3.csv", delimiter=',')
	X3_HomoHit=np.matrix(m3_HomoHit)
	m4_HomoHit = np.loadtxt("Homo/Homo-Hit-NotSrc-PerSec-4.csv", delimiter=',')
	X4_HomoHit=np.matrix(m4_HomoHit)
	m5_HomoHit = np.loadtxt("Homo/Homo-Hit-NotSrc-PerSec-5.csv", delimiter=',')
	X5_HomoHit=np.matrix(m5_HomoHit)
	m6_HomoHit = np.loadtxt("Homo/Homo-Hit-NotSrc-PerSec-6.csv", delimiter=',')
	X6_HomoHit=np.matrix(m6_HomoHit)
	m7_HomoHit = np.loadtxt("Homo/Homo-Hit-NotSrc-PerSec-7.csv", delimiter=',')
	X7_HomoHit=np.matrix(m7_HomoHit)
	m8_HomoHit = np.loadtxt("Homo/Homo-Hit-NotSrc-PerSec-8.csv", delimiter=',')
	X8_HomoHit=np.matrix(m8_HomoHit)
	m9_HomoHit = np.loadtxt("Homo/Homo-Hit-NotSrc-PerSec-9.csv", delimiter=',')
	X9_HomoHit=np.matrix(m9_HomoHit)
	m10_HomoHit = np.loadtxt("Homo/Homo-Hit-NotSrc-PerSec-10.csv", delimiter=',')
	X10_HomoHit=np.matrix(m10_HomoHit)'''
	
	m_mean_HomoHit = np.zeros((23,600))	
	
	m1_HomoHitMiss = np.loadtxt("Homo/Homo-HitMiss-NotSrc-PerSec-1.csv", delimiter=',')
	X1_HomoHitMiss=np.matrix(m1_HomoHitMiss)
	'''m2_HomoHitMiss = np.loadtxt("Homo/Homo-HitMiss-NotSrc-PerSec-2.csv", delimiter=',')
	X2_HomoHitMiss=np.matrix(m2_HomoHitMiss)
	m3_HomoHitMiss = np.loadtxt("Homo/Homo-HitMiss-NotSrc-PerSec-3.csv", delimiter=',')
	X3_HomoHitMiss=np.matrix(m3_HomoHitMiss)
	m4_HomoHitMiss = np.loadtxt("Homo/Homo-HitMiss-NotSrc-PerSec-4.csv", delimiter=',')
	X4_HomoHitMiss=np.matrix(m4_HomoHitMiss)
	m5_HomoHitMiss = np.loadtxt("Homo/Homo-HitMiss-NotSrc-PerSec-5.csv", delimiter=',')
	X5_HomoHitMiss=np.matrix(m5_HomoHitMiss)
	m6_HomoHitMiss = np.loadtxt("Homo/Homo-HitMiss-NotSrc-PerSec-6.csv", delimiter=',')
	X6_HomoHitMiss=np.matrix(m6_HomoHitMiss)
	m7_HomoHitMiss = np.loadtxt("Homo/Homo-HitMiss-NotSrc-PerSec-7.csv", delimiter=',')
	X7_HomoHitMiss=np.matrix(m7_HomoHitMiss)
	m8_HomoHitMiss = np.loadtxt("Homo/Homo-HitMiss-NotSrc-PerSec-8.csv", delimiter=',')
	X8_HomoHitMiss=np.matrix(m8_HomoHitMiss)
	m9_HomoHitMiss = np.loadtxt("Homo/Homo-HitMiss-NotSrc-PerSec-9.csv", delimiter=',')
	X9_HomoHitMiss=np.matrix(m9_HomoHitMiss)
	m10_HomoHitMiss = np.loadtxt("Homo/Homo-HitMiss-NotSrc-PerSec-10.csv", delimiter=',')
	X10_HomoHitMiss=np.matrix(m10_HomoHitMiss)'''
	
	m_mean_HomoHitMiss = np.zeros((23,600))
	HitRateHomo=np.zeros((23,600))

	m1_PCAHit = np.loadtxt("PCA-With Betweenness/PCA-Hit-NotSrc-PerSec-1.csv", delimiter=',')
	X1_PCAHit=np.matrix(m1_PCAHit)
	'''m2_PCAHit = np.loadtxt("PCA-With Betweenness/PCA-Hit-NotSrc-PerSec-2.csv", delimiter=',')
	X2_PCAHit=np.matrix(m2_PCAHit)
	m3_PCAHit = np.loadtxt("PCA-With Betweenness/PCA-Hit-NotSrc-PerSec-3.csv", delimiter=',')
	X3_PCAHit=np.matrix(m3_PCAHit)
	m4_PCAHit = np.loadtxt("PCA-With Betweenness/PCA-Hit-NotSrc-PerSec-4.csv", delimiter=',')
	X4_PCAHit=np.matrix(m4_PCAHit)
	m5_PCAHit = np.loadtxt("PCA-With Betweenness/PCA-Hit-NotSrc-PerSec-5.csv", delimiter=',')
	X5_PCAHit=np.matrix(m5_PCAHit)
	m6_PCAHit = np.loadtxt("PCA-With Betweenness/PCA-Hit-NotSrc-PerSec-6.csv", delimiter=',')
	X6_PCAHit=np.matrix(m6_PCAHit)
	m7_PCAHit = np.loadtxt("PCA-With Betweenness/PCA-Hit-NotSrc-PerSec-7.csv", delimiter=',')
	X7_PCAHit=np.matrix(m7_PCAHit)
	m8_PCAHit = np.loadtxt("PCA-With Betweenness/PCA-Hit-NotSrc-PerSec-8.csv", delimiter=',')
	X8_PCAHit=np.matrix(m8_PCAHit)
	m9_PCAHit = np.loadtxt("PCA-With Betweenness/PCA-Hit-NotSrc-PerSec-9.csv", delimiter=',')
	X9_PCAHit=np.matrix(m9_PCAHit)
	m10_PCAHit = np.loadtxt("PCA-With Betweenness/PCA-Hit-NotSrc-PerSec-10.csv", delimiter=',')
	X10_PCAHit=np.matrix(m10_PCAHit)'''
	
	m_mean_PCAHit = np.zeros((23,600))	
	m1_PCAHitMiss = np.loadtxt("PCA-With Betweenness/PCA-HitMiss-NotSrc-PerSec-1.csv", delimiter=',')
	X1_PCAHitMiss=np.matrix(m1_PCAHitMiss)
	'''m2_PCAHitMiss = np.loadtxt("PCA-With Betweenness/PCA-HitMiss-NotSrc-PerSec-2.csv", delimiter=',')
	X2_PCAHitMiss=np.matrix(m2_PCAHitMiss)
	m3_PCAHitMiss = np.loadtxt("PCA-With Betweenness/PCA-HitMiss-NotSrc-PerSec-3.csv", delimiter=',')
	X3_PCAHitMiss=np.matrix(m3_PCAHitMiss)
	m4_PCAHitMiss = np.loadtxt("PCA-With Betweenness/PCA-HitMiss-NotSrc-PerSec-4.csv", delimiter=',')
	X4_PCAHitMiss=np.matrix(m4_PCAHitMiss)
	m5_PCAHitMiss = np.loadtxt("PCA-With Betweenness/PCA-HitMiss-NotSrc-PerSec-5.csv", delimiter=',')
	X5_PCAHitMiss=np.matrix(m5_PCAHitMiss)
	m6_PCAHitMiss = np.loadtxt("PCA-With Betweenness/PCA-HitMiss-NotSrc-PerSec-6.csv", delimiter=',')
	X6_PCAHitMiss=np.matrix(m6_PCAHitMiss)
	m7_PCAHitMiss = np.loadtxt("PCA-With Betweenness/PCA-HitMiss-NotSrc-PerSec-7.csv", delimiter=',')
	X7_PCAHitMiss=np.matrix(m7_PCAHitMiss)
	m8_PCAHitMiss = np.loadtxt("PCA-With Betweenness/PCA-HitMiss-NotSrc-PerSec-8.csv", delimiter=',')
	X8_PCAHitMiss=np.matrix(m8_PCAHitMiss)
	m9_PCAHitMiss = np.loadtxt("PCA-With Betweenness/PCA-HitMiss-NotSrc-PerSec-9.csv", delimiter=',')
	X9_PCAHitMiss=np.matrix(m9_PCAHitMiss)
	m10_PCAHitMiss = np.loadtxt("PCA-With Betweenness/PCA-HitMiss-NotSrc-PerSec-10.csv", delimiter=',')
	X10_PCAHitMiss=np.matrix(m10_PCAHitMiss)'''
	
	m_mean_PCAHitMiss = np.zeros((23,600))
	HitRatePCA=np.zeros((23,600))

	m1_DegreeHit = np.loadtxt("Deg/Deg-Hit-NotSrc-PerSec-1.csv", delimiter=',')
	X1_DegreeHit=np.matrix(m1_DegreeHit)
	'''m2_DegreeHit = np.loadtxt("Deg/Deg-Hit-NotSrc-PerSec-2.csv", delimiter=',')
	X2_DegreeHit=np.matrix(m2_DegreeHit)
	m3_DegreeHit = np.loadtxt("Deg/Deg-Hit-NotSrc-PerSec-3.csv", delimiter=',')
	X3_DegreeHit=np.matrix(m3_DegreeHit)
	m4_DegreeHit = np.loadtxt("Deg/Deg-Hit-NotSrc-PerSec-4.csv", delimiter=',')
	X4_DegreeHit=np.matrix(m4_DegreeHit)
	m5_DegreeHit = np.loadtxt("Deg/Deg-Hit-NotSrc-PerSec-5.csv", delimiter=',')
	X5_DegreeHit=np.matrix(m5_DegreeHit)
	m6_DegreeHit = np.loadtxt("Deg/Deg-Hit-NotSrc-PerSec-6.csv", delimiter=',')
	X6_DegreeHit=np.matrix(m6_DegreeHit)
	m7_DegreeHit = np.loadtxt("Deg/Deg-Hit-NotSrc-PerSec-7.csv", delimiter=',')
	X7_DegreeHit=np.matrix(m7_DegreeHit)
	m8_DegreeHit = np.loadtxt("Deg/Deg-Hit-NotSrc-PerSec-8.csv", delimiter=',')
	X8_DegreeHit=np.matrix(m8_DegreeHit)
	m9_DegreeHit = np.loadtxt("Deg/Deg-Hit-NotSrc-PerSec-9.csv", delimiter=',')
	X9_DegreeHit=np.matrix(m9_DegreeHit)
	m10_DegreeHit = np.loadtxt("Deg/Deg-Hit-NotSrc-PerSec-10.csv", delimiter=',')
	X10_DegreeHit=np.matrix(m10_DegreeHit)'''
	
	m_mean_DegreeHit = np.zeros((23,600))	
	
	m1_DegreeHitMiss = np.loadtxt("Deg/Deg-HitMiss-NotSrc-PerSec-1.csv", delimiter=',')
	X1_DegreeHitMiss=np.matrix(m1_DegreeHitMiss)
	'''m2_DegreeHitMiss = np.loadtxt("Deg/Deg-HitMiss-NotSrc-PerSec-2.csv", delimiter=',')
	X2_DegreeHitMiss=np.matrix(m2_DegreeHitMiss)
	m3_DegreeHitMiss = np.loadtxt("Deg/Deg-HitMiss-NotSrc-PerSec-3.csv", delimiter=',')
	X3_DegreeHitMiss=np.matrix(m3_DegreeHitMiss)
	m4_DegreeHitMiss = np.loadtxt("Deg/Deg-HitMiss-NotSrc-PerSec-4.csv", delimiter=',')
	X4_DegreeHitMiss=np.matrix(m4_DegreeHitMiss)
	m5_DegreeHitMiss = np.loadtxt("Deg/Deg-HitMiss-NotSrc-PerSec-5.csv", delimiter=',')
	X5_DegreeHitMiss=np.matrix(m5_DegreeHitMiss)
	m6_DegreeHitMiss = np.loadtxt("Deg/Deg-HitMiss-NotSrc-PerSec-6.csv", delimiter=',')
	X6_DegreeHitMiss=np.matrix(m6_DegreeHitMiss)
	m7_DegreeHitMiss = np.loadtxt("Deg/Deg-HitMiss-NotSrc-PerSec-7.csv", delimiter=',')
	X7_DegreeHitMiss=np.matrix(m7_DegreeHitMiss)
	m8_DegreeHitMiss = np.loadtxt("Deg/Deg-HitMiss-NotSrc-PerSec-8.csv", delimiter=',')
	X8_DegreeHitMiss=np.matrix(m8_DegreeHitMiss)
	m9_DegreeHitMiss = np.loadtxt("Deg/Deg-HitMiss-NotSrc-PerSec-9.csv", delimiter=',')
	X9_DegreeHitMiss=np.matrix(m9_DegreeHitMiss)
	m10_DegreeHitMiss = np.loadtxt("Deg/Deg-HitMiss-NotSrc-PerSec-10.csv", delimiter=',')
	X10_DegreeHitMiss=np.matrix(m10_DegreeHitMiss)'''
	
	m_mean_DegreeHitMiss = np.zeros((23,600))
	HitRateDegree=np.zeros((23,600))
	for i in range(m_mean_PCAHitMiss.shape[0]):
		for j in range(m_mean_PCAHitMiss.shape[1]):
			
			m_mean_HomoHit.itemset((i,j),(X1_HomoHit.item(i,j)))
			#+X2_HomoHit.item(i,j)+X3_HomoHit.item(i,j)+X4_HomoHit.item(i,j)+X5_HomoHit.item(i,j)+X6_HomoHit.item(i,j)+X7_HomoHit.item(i,j)+X8_HomoHit.item(i,j)+X9_HomoHit.item(i,j)+X10_HomoHit.item(i,j))/10)
			m_mean_HomoHitMiss.itemset((i,j),(X1_HomoHitMiss.item(i,j)))
			#+X2_HomoHitMiss.item(i,j)+X3_HomoHitMiss.item(i,j)+X4_HomoHitMiss.item(i,j)+X5_HomoHitMiss.item(i,j)+X6_HomoHitMiss.item(i,j)+X7_HomoHitMiss.item(i,j)+X8_HomoHitMiss.item(i,j)+X9_HomoHitMiss.item(i,j)+X10_HomoHitMiss.item(i,j))/10)
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

	font = {'family' : 'normal',        	'weight' : 'bold',        	'size'   : 24}

	matplotlib.rc('font', **font)
	tedaad = 6
	x = np.linspace(0,tedaad,tedaad)
	#AVG_NoCache = np.zeros((1,600))
	#AVG_NoCache = np.sum(HitRateNoCache, axis=0)/11
	AVG_Homo = np.zeros((1,600))
	#print np.sum(HitRateHomo[12:23,0:600], axis=0).shape
	
	AVG_Homo = np.sum(HitRateHomo[12:23,0:600], axis=0)/11
	#print AVG_Homo
	AVG_Homo_Imp = np.zeros((1,600))
	AVG_Homo_Imp = (HitRateHomo[12,0:600]+HitRateHomo[14,0:600]+HitRateHomo[16,0:600]+HitRateHomo[17,0:600]+HitRateHomo[18,0:600]+HitRateHomo[20,0:600]+HitRateHomo[21,0:600]+HitRateHomo[22,0:600])/8
	#print AVG_Homo.shape
	AVG_Homo_NotImp = np.zeros((1,600))
	AVG_Homo_NotImp = (HitRateHomo[13,0:600]+HitRateHomo[15,0:600]+HitRateHomo[19,0:600])/3

	AVG_Degree = np.zeros((1,600))
	AVG_Degree = np.sum(HitRateDegree[12:23,0:600], axis=0)/11	

	AVG_Degree_Imp = np.zeros((1,600))
	AVG_Degree_Imp = (HitRateDegree[12,0:600]+HitRateDegree[14,0:600]+HitRateDegree[16,0:600]+HitRateDegree[17,0:600]+HitRateDegree[18,0:600]+HitRateDegree[20,0:600]+HitRateDegree[21,0:600]+HitRateDegree[22,0:600])/8	

	AVG_Degree_NotImp = np.zeros((1,600))
	AVG_Degree_NotImp = (HitRateDegree[13,0:600]+HitRateDegree[15,0:600]+HitRateDegree[19,0:600])/3	

	AVG_PCA = np.zeros((1,600))
	AVG_PCA = np.sum(HitRatePCA[12:23,0:600], axis=0)/11	

	AVG_PCA_Imp = np.zeros((1,600))
	AVG_PCA_Imp = (HitRatePCA[12,0:600]+HitRatePCA[14,0:600]+HitRatePCA[16,0:600]+HitRatePCA[17,0:600]+HitRatePCA[18,0:600]+HitRatePCA[20,0:600]+HitRatePCA[21,0:600]+HitRatePCA[22,0:600])/8	
	
	AVG_PCA_NotImp = np.zeros((1,600))
	AVG_PCA_NotImp = (HitRatePCA[13,0:600]+HitRatePCA[15,0:600]+HitRatePCA[19,0:600])/3

	counter = 0
	for i in range(AVG_PCA.shape[0]):
		if AVG_PCA[i] > AVG_Degree[i] or AVG_PCA[i] == AVG_Degree[i] or AVG_PCA[i]> AVG_Homo[i] or AVG_PCA[i] == AVG_Homo[i]:
			counter = counter + 1

	#print np.around(np.average((AVG_Homo)),6),np.around(np.average((AVG_Degree)),6),np.around(np.average((AVG_PCA)),6)
	#print (HitRateHomo[12:23,:]).shape
	print np.around(np.average((HitRateHomo[12:23,:])),6),"\n",np.around(np.average((HitRateDegree[12:23,:])),6),"\n",np.around(np.average((HitRatePCA[12:23,:])),6),"\n"
	
	#print np.around(np.average((np.sum(HitRateHomo[12:23,0:600], axis=0)/11)),6),np.around(np.average((np.sum(HitRateDegree[12:23,0:600], axis=0)/11)),6),np.around(np.average((np.sum(HitRatePCA[12:23,0:600], axis=0)/11)),6)	
	print counter
	
	#print np.around(np.average((AVG_Homo_Imp)),6),np.around(np.average((AVG_Degree_Imp)),6),np.around(np.average((AVG_PCA_Imp)),6)
	print np.around(np.average((HitRateHomo[12,:]+HitRateHomo[14,:]+HitRateHomo[17,:]+HitRateHomo[18,:]+HitRateHomo[21,:]+HitRateHomo[22,:])/6),6),np.around(np.average((HitRateDegree[12,:]+HitRateDegree[14,:]+HitRateDegree[17,:]+HitRateDegree[18,:]+HitRateDegree[21,:]+HitRateDegree[22,:])/6),6),np.around(np.average((HitRatePCA[12,:]+HitRatePCA[14,:]+HitRatePCA[17,:]+HitRatePCA[18,:]+HitRatePCA[21,:]+HitRatePCA[22,:])/6),6)
	#print np.around(np.average((AVG_Homo_NotImp)),6),np.around(np.average((AVG_Degree_NotImp)),6),np.around(np.average((AVG_PCA_NotImp)),6)
	print np.around(np.average((HitRateHomo[13,:]+HitRateHomo[15,:]+HitRateHomo[16,:]+HitRateHomo[19,:]+HitRateHomo[20,:])/5),6),np.around(np.average((HitRateDegree[13,:]+HitRateDegree[15,:]+HitRateDegree[16,:]+HitRateDegree[19,:]+HitRateDegree[20,:])/5),6),np.around(np.average((HitRatePCA[13,:]+HitRatePCA[15,:]+HitRatePCA[16,:]+HitRatePCA[19,:]+HitRatePCA[20,:])/5),6)

	#print np.around(np.average(HitRateHomo[12,:]+HitRateHomo[13,:]+HitRateHomo[14,:]+HitRateHomo[15,:]+HitRateHomo[16,:]+HitRateHomo[17,:]+HitRateHomo[18,:]+HitRateHomo[19,:]+HitRateHomo[20,:]+HitRateHomo[21,:]+HitRateHomo[22,:]),6),np.around(np.average(HitRateDegree[12,:]+HitRateDegree[13,:]+HitRateDegree[14,:]+HitRateDegree[15,:]+HitRateDegree[16,:]+HitRateDegree[17,:]+HitRateDegree[18,:]+HitRateDegree[19,:]+HitRateDegree[20,:]+HitRateDegree[21,:]+HitRateDegree[22,:]),6),np.around(np.average(HitRatePCA[12,:]+HitRatePCA[13,:]+HitRatePCA[14,:]+HitRatePCA[15,:]+HitRatePCA[16,:]+HitRatePCA[17,:]+HitRatePCA[18,:]+HitRatePCA[19,:]+HitRatePCA[20,:]+HitRatePCA[21,:]+HitRatePCA[22,:]),6)

	#print np.around(np.average(((AVG_Homo_Imp+AVG_Homo_NotImp)/2)),6),np.around(np.average(((AVG_Degree_Imp+AVG_Degree_NotImp)/2)),6),np.around(np.average(((AVG_PCA_Imp+AVG_PCA_NotImp)/2)),6)

	Smooth_Homo = np.zeros((tedaad,1))#(np.zeros((1,tedaad)))
	Smooth_Degree = np.zeros((tedaad,1))#(np.zeros((1,tedaad)))
	Smooth_PCA = np.zeros((tedaad,1))#(np.zeros((1,tedaad)))
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
	#Smooth_Homo = (Smooth_Homo)
	#Smooth_Degree = (Smooth_Degree)
	#Smooth_PCA = (Smooth_PCA)
	print Smooth_Homo
	print Smooth_Degree
	print Smooth_PCA	
	#plt.plot(x,(AVG_NoCache),'o-g',label="NoCache")
	plt.plot(x,Smooth_Homo,'.-g',label="Uniform")
	plt.plot(x,Smooth_Degree,'o-b',label="Degree")
	plt.plot(x,Smooth_PCA,'^-r',label="ProposedMethod")

	plt.legend(loc='upper left')
	plt.title('Comparison in Core Rtrs Hit ratio')
	plt.xlim([0,tedaad])
	plt.ylim([0.00,0.015])
	plt.xlabel('Simulation Time (*100 Seconds)')
	plt.ylabel('HitRatio')
	plt.show()

	#AVG_NoCache = np.zeros((1,600))
	#AVG_NoCache = np.sum(HitRateNoCache, axis=0)/11
	#AVG_Homo = np.zeros((1,600))
	#AVG_Homo = np.sum(HitRateHomo[0:12,0:600], axis=0)/12
	#AVG_Degree = np.zeros((1,600))
	#AVG_Degree = np.sum(HitRateDegree[0:12,0:600], axis=0)/12
	#AVG_PCA = np.zeros((1,600))
	#AVG_PCA = np.sum(HitRatePCA[0:12,0:600], axis=0)/12	
	#print np.around(np.average((AVG_Homo)),6),np.around(np.average((AVG_Degree)),6),np.around(np.average((AVG_PCA)),6)
	#counter = 0
	#for i in range(AVG_PCA.shape[0]):
		#if AVG_PCA[i] == AVG_Degree[i]:
			#counter = counter + 1	
	#print counter
	#x2 = np.linspace(0,600,600)
	#print np.average((AVG_Homo)),np.average((AVG_Degree)),np.average((AVG_PIT)), np.average((AVG_CS)),np.average((AVG_PCA))
	#plt.plot(x,(AVG_NoCache),'o-g',label="NoCache")
	#plt.plot(x,(AVG_Homo),'.-g',label="Homo")
	#plt.plot(x2,(AVG_Degree),'o-b',label="Degree")
	#plt.plot(x2,(AVG_PCA),'^-r',label="ProposedMethod")
	#plt.legend(loc='upper right')
	#plt.title('Comparison in Consumer Rtr load')
	#plt.xlim([0,600])
	#plt.ylim([0,0.03])
	#plt.xlabel('Time(Second)')
	#plt.ylabel('HitRatio')
	#plt.show()
np.array(CS(),dtype='float32')
#fig = plt.figure()
#plt.plot(data)
#fig.suptitle('test title')
#plt.xlabel('xlabel')
#plt.ylabel('ylabel')
#fig.savefig('test.jpg')
