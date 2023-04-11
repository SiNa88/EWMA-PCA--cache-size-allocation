import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import matplotlib.cm as cm
from pylab import *
from matplotlib.font_manager import FontProperties


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

	m1Homo = np.loadtxt("Rcvd-Requests-Homo-PerSec-1.csv", delimiter=',')
	X1Homo=np.matrix(m1Homo)
	m2Homo = np.loadtxt("Rcvd-Requests-Homo-PerSec-2.csv", delimiter=',')
	X2Homo=np.matrix(m2Homo)
	m3Homo = np.loadtxt("Rcvd-Requests-Homo-PerSec-3.csv", delimiter=',')
	X3Homo=np.matrix(m3Homo)
	m4Homo = np.loadtxt("Rcvd-Requests-Homo-PerSec-4.csv", delimiter=',')
	X4Homo=np.matrix(m4Homo)
	m5Homo = np.loadtxt("Rcvd-Requests-Homo-PerSec-5.csv", delimiter=',')
	X5Homo=np.matrix(m5Homo)
	m6Homo = np.loadtxt("Rcvd-Requests-Homo-PerSec-6.csv", delimiter=',')
	X6Homo=np.matrix(m6Homo)
	m7Homo = np.loadtxt("Rcvd-Requests-Homo-PerSec-7.csv", delimiter=',')
	X7Homo=np.matrix(m7Homo)
	m8Homo = np.loadtxt("Rcvd-Requests-Homo-PerSec-8.csv", delimiter=',')
	X8Homo=np.matrix(m8Homo)
	m9Homo = np.loadtxt("Rcvd-Requests-Homo-PerSec-9.csv", delimiter=',')
	X9Homo=np.matrix(m9Homo)
	m10Homo = np.loadtxt("Rcvd-Requests-Homo-PerSec-10.csv", delimiter=',')
	X10Homo=np.matrix(m10Homo)
	m_meanHomo = np.zeros((11,600))
	
	m1Deg = np.loadtxt("Rcvd-Requests-Deg-PerSec-1.csv", delimiter=',')
	X1Deg=np.matrix(m1Deg)
	m2Deg = np.loadtxt("Rcvd-Requests-Deg-PerSec-2.csv", delimiter=',')
	X2Deg=np.matrix(m2Deg)
	m3Deg = np.loadtxt("Rcvd-Requests-Deg-PerSec-3.csv", delimiter=',')
	X3Deg=np.matrix(m3Deg)
	m4Deg = np.loadtxt("Rcvd-Requests-Deg-PerSec-4.csv", delimiter=',')
	X4Deg=np.matrix(m4Deg)
	m5Deg = np.loadtxt("Rcvd-Requests-Deg-PerSec-5.csv", delimiter=',')
	X5Deg=np.matrix(m5Deg)
	m6Deg = np.loadtxt("Rcvd-Requests-Deg-PerSec-6.csv", delimiter=',')
	X6Deg=np.matrix(m6Deg)
	m7Deg = np.loadtxt("Rcvd-Requests-Deg-PerSec-7.csv", delimiter=',')
	X7Deg=np.matrix(m7Deg)
	m8Deg = np.loadtxt("Rcvd-Requests-Deg-PerSec-8.csv", delimiter=',')
	X8Deg=np.matrix(m8Deg)
	m9Deg = np.loadtxt("Rcvd-Requests-Deg-PerSec-9.csv", delimiter=',')
	X9Deg=np.matrix(m9Deg)
	m10Deg = np.loadtxt("Rcvd-Requests-Deg-PerSec-10.csv", delimiter=',')
	X10Deg=np.matrix(m10Deg)
	m_meanDeg = np.zeros((11,600))

	for i in range(m_mean.shape[0]):
		for j in range(m_mean.shape[1]):
			m_mean.itemset((i,j),(X1.item(i,j)+X2.item(i,j)+X3.item(i,j)+X4.item(i,j)+X5.item(i,j)+X6.item(i,j)+X7.item(i,j)+X8.item(i,j)+X9.item(i,j)+X10.item(i,j))/10)
	for i in range(m_meanHomo.shape[0]):
		for j in range(m_meanHomo.shape[1]):
			m_meanHomo.itemset((i,j),(X1Homo.item(i,j)+X2Homo.item(i,j)+X3Homo.item(i,j)+X4Homo.item(i,j)+X5Homo.item(i,j)+X6Homo.item(i,j)+X7Homo.item(i,j)+X8Homo.item(i,j)+X9Homo.item(i,j)+X10Homo.item(i,j))/10)
			m_meanDeg.itemset((i,j),(X1Deg.item(i,j)+X2Deg.item(i,j)+X3Deg.item(i,j)+X4Deg.item(i,j)+X5Deg.item(i,j)+X6Deg.item(i,j)+X7Deg.item(i,j)+X8Deg.item(i,j)+X9Deg.item(i,j)+X10Deg.item(i,j))/10)

	AVG_Homo = np.zeros((1,600))	
	AVG_Homo = np.sum(m_meanHomo, axis=0)/11

	AVG_Deg = np.zeros((1,600))	
	AVG_Deg = np.sum(m_meanDeg, axis=0)/11
	X1_1stRow = m_mean[8,:].transpose()
	X12_1stRow = m_mean[12,:].transpose()
	X13_1stRow = m_mean[13,:].transpose()
	X14_1stRow = m_mean[14,:].transpose()
	X15_1stRow = m_mean[15,:].transpose()
	X16_1stRow = m_mean[16,:].transpose()
	X17_1stRow = m_mean[17,:].transpose()
	X18_1stRow = m_mean[18,:].transpose()
	X19_1stRow = m_mean[19,:].transpose()
	X20_1stRow = m_mean[20,:].transpose()
	X21_1stRow = m_mean[21,:].transpose()
	X22_1stRow = m_mean[22,:].transpose()
	tedaad = 10
	x_axis=np.linspace(0,tedaad,tedaad)
	
	font = {'family' : 'normal',
        	'weight' : 'bold',
        	'size'   : 24}

	matplotlib.rc('font', **font)
	
	Smooth1 = np.zeros((tedaad,1))
	Smooth12 = np.zeros((tedaad,1))
	Smooth13 = np.zeros((tedaad,1))
	Smooth14 = np.zeros((tedaad,1))
	Smooth15 = np.zeros((tedaad,1))
	Smooth16 = np.zeros((tedaad,1))
	Smooth17 = np.zeros((tedaad,1))
	Smooth18 = np.zeros((tedaad,1))
	Smooth19 = np.zeros((tedaad,1))
	Smooth20 = np.zeros((tedaad,1))
	Smooth21 = np.zeros((tedaad,1))
	Smooth22 = np.zeros((tedaad,1))
	Smooth_Homo = np.zeros((tedaad,1))
	Smooth_Deg = np.zeros((tedaad,1))
	j = 0
	for i in range(X12_1stRow.shape[0]):
		if ((j+1)*(600/tedaad)) <= i:
			j = j + 1
		Smooth1[j] = Smooth1[j] + X1_1stRow[i]		
		Smooth12[j] = Smooth12[j] + X12_1stRow[i]
		Smooth13[j] = Smooth13[j] + X13_1stRow[i]
		Smooth14[j] = Smooth14[j] + X14_1stRow[i]
		Smooth15[j] = Smooth15[j] + X15_1stRow[i]
		Smooth16[j] = Smooth16[j] + X16_1stRow[i]
		Smooth17[j] = Smooth17[j] + X17_1stRow[i]
		Smooth18[j] = Smooth18[j] + X18_1stRow[i]
		Smooth19[j] = Smooth19[j] + X19_1stRow[i]
		Smooth20[j] = Smooth20[j] + X20_1stRow[i]
		Smooth21[j] = Smooth21[j] + X21_1stRow[i]
		Smooth22[j] = Smooth22[j] + X22_1stRow[i]
		Smooth_Homo[j] = Smooth_Homo[j] + AVG_Homo[i]
		Smooth_Deg[j] = Smooth_Deg[j] + AVG_Deg[i]
	for i in range(Smooth12.shape[0]):
		Smooth1[i] = Smooth1[i] / (600/tedaad)		
		Smooth12[i] = Smooth12[i] / (600/tedaad)
		Smooth13[i] = Smooth13[i] / (600/tedaad)
		Smooth14[i] = Smooth14[i] / (600/tedaad)
		Smooth15[i] = Smooth15[i] / (600/tedaad)
		Smooth16[i] = Smooth16[i] / (600/tedaad)
		Smooth17[i] = Smooth17[i] / (600/tedaad)
		Smooth18[i] = Smooth18[i] / (600/tedaad)
		Smooth19[i] = Smooth19[i] / (600/tedaad)
		Smooth20[i] = Smooth20[i] / (600/tedaad)
		Smooth21[i] = Smooth21[i] / (600/tedaad)
		Smooth22[i] = Smooth22[i] / (600/tedaad)
		Smooth_Homo[i] = Smooth_Homo[i] / (600/tedaad)
		Smooth_Deg[i] = Smooth_Deg[i] / (600/tedaad)
	Smooth_mean = np.zeros((tedaad,1))
	for i in range(Smooth12.shape[0]):
		Smooth_mean[i] =(Smooth12[i]+Smooth13[i]+Smooth14[i]+Smooth15[i]+Smooth16[i]+Smooth17[i]+Smooth18[i]+Smooth19[i]+Smooth20[i]+Smooth21[i]+Smooth22[i])/11
	#Smooth = np.copy(Smooth)
	#print Smooth12
	# 22
	a = (5/(x_axis**0.85+7.8))+0.2
	#RTR7--> a = (6/(x_axis**1+10.2))+0.37
	#RTR0--> a = (6/(x_axis**1+4.7))+1.54

	#plt.scatter(x_axis,Smooth12)
	#plt.scatter(x_axis,Smooth13)
	#plt.scatter(x_axis,Smooth14)
	#plt.scatter(x_axis,Smooth15)
	#plt.scatter(x_axis,Smooth16)
	#plt.scatter(x_axis,Smooth17)
	#plt.scatter(x_axis,Smooth18)
	#plt.scatter(x_axis,Smooth19)
	#plt.scatter(x_axis,Smooth20)
	#plt.scatter(x_axis,Smooth21)
	plt.plot(x_axis,a,'.-b',label="Theory")
	#fontP = FontProperties()
	#fontP.set_size('small')
	#legend([plot1], "title")
	#plt.plot(x_axis,Smooth12,'^-b',label="Rtr0")
	#plt.plot(x_axis,Smooth13,'^-g',label="Rtr1")
	#plt.plot(x_axis,Smooth14,'^-r',label="Rtr2")
	#plt.plot(x_axis,Smooth15,'^-c',label="Rtr3")
	#plt.plot(x_axis,Smooth16,'^-m',label="Rtr4")
	#plt.plot(x_axis,Smooth17,'^-y',label="Rtr5")
	#plt.plot(x_axis,Smooth18,'^-k',label="Rtr6")
	#plt.plot(x_axis,Smooth19,'o-b',label="Rtr7")
	plt.plot(x_axis,Smooth1,'^-r',label="Simulation")
	#plt.plot(x_axis,Smooth21,'o-r',label="Rtr9")
	#plt.plot(x_axis,Smooth22,'o-c',label="Rtr10")
	plt.xlim([0,tedaad+0.1])
	plt.ylim([0.5,0.86])
	#RTR7-->plt.ylim([0.6,1])
	#Rtr0-->plt.ylim([1.9,2.9])
	plt.xlabel('Time(*60 Seconds)')
	plt.ylabel('Pending Requests')
	plt.legend(loc='upper right',fancybox=True)
	#plt.legend(loc='upper left',bbox_to_anchor=(0.5, 1),ncol=1, fancybox=True)
	plt.title('SnapShot of Pending Interest Tables in Consumer9')
	#plt.title('SnapShot of Pending Interest Tables in Rtr7')
	plt.show()
	
MyStem()
