#%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('seaborn-white')
import math
import find_class

# this fuction find the Gassian of a given class .it is used for contour plotting of geven class
def G(pos,mu,inv_cov_matrix):        
	n = 2
	Sigma_det = np.linalg.det(np.linalg.inv(inv_cov_matrix))
	N = np.sqrt((2*np.pi)**n * Sigma_det)

	# This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
	# way across all the input variables.
	fac = np.einsum('...k,kl,...l->...', pos-mu, inv_cov_matrix, pos-mu)

	return np.exp(-fac / 2) / N


# this function plot the data of all the 3 classes with decion boundary
def plot_1(p_c1,p_c2,p_c3,inv_cov_matrix1,inv_cov_matrix2,inv_cov_matrix3,mean1,mean2,mean3,x1,x2,x3,y1,y2,y3,which_plot,c_min_max):
    
	x=np.zeros(2)
	x11=np.zeros(1000000)
	x22=np.zeros(1000000)
	x33=np.zeros(1000000)
	y11=np.zeros(1000000)
	y22=np.zeros(1000000)
	y33=np.zeros(1000000)
	minx=min(c_min_max[0],c_min_max[4],c_min_max[8])
	maxx=max(c_min_max[1],c_min_max[5],c_min_max[9])
	miny=min(c_min_max[2],c_min_max[6],c_min_max[10])
	maxy=max(c_min_max[3],c_min_max[7],c_min_max[11])
	l1=0
	l2=0
	l3=0
	x_intr=(maxx-minx)/250.00;
	y_intr=(maxy-miny)/250.00;
	i=minx-(0.05*(maxx-minx))
	while i < maxx+(0.05*(maxx-minx)):
                j=miny-(0.05*(maxy-miny))
		while j < maxy+(0.05*(maxy-miny)):
			x[0]=i
			x[1]=j
			clas=find_class.find_class_case(p_c1,p_c2,p_c3,inv_cov_matrix1,inv_cov_matrix2,inv_cov_matrix3,mean1,mean2,mean3,x)
			if clas==0:
				x11[l1]=i
				y11[l1]=j
				l1=l1+1
			elif clas==1:	
				x22[l2]=i
				y22[l2]=j
				l2=l2+1
			elif clas==2:
				x33[l3]=i
				y33[l3]=j
				l3=l3+1
                        j=j+y_intr
                i=i+x_intr

    
	if which_plot==0:
		plt.plot(x11,y11,'#DEB887',label='class1_predicted')
		plt.plot(x22,y22,'#53868B',label='class2_predicted')
		plt.plot(x33,y33,'#458B00',label='class3_predicted')

		plt.plot(x1,y1,'ro',label='class1_data')
		plt.plot(x2,y2,'bo',label='class2_data')
		plt.plot(x3,y3,'go',label='class3_data')
		plt.legend()
		plt.show()
	# plot the contour of all the 3 classes	
	elif which_plot==10:
		c1_pos=np.zeros((1000,1000,1000))
		c1_x = np.linspace(c_min_max[0],c_min_max[1], 100) 
		c1_y = np.linspace(c_min_max[2],c_min_max[3], 100)
		c1_X, c1_Y = np.meshgrid(c1_x,c1_y)    #
		c1_pos = np.empty(c1_X.shape + (2,))  
		c1_pos[:, :, 0] = c1_X
		c1_pos[:, :, 1] = c1_Y
		c1_Z =G(c1_pos,mean1,inv_cov_matrix1)   

		c2_pos=np.zeros((1000,1000,1000))
		c2_x = np.linspace(c_min_max[4],c_min_max[5], 100)
		c2_y = np.linspace(c_min_max[6],c_min_max[7] ,100)
		c2_X, c2_Y = np.meshgrid(c2_x,c2_y)
		c2_pos = np.empty(c2_X.shape + (2,))
		c2_pos[:, :, 0] = c2_X
		c2_pos[:, :, 1] = c2_Y
		c2_Z =G(c2_pos,mean2,inv_cov_matrix2)

		c3_pos=np.zeros((1000,1000,1000))
		c3_x = np.linspace(c_min_max[8],c_min_max[9], 100)
		c3_y = np.linspace(c_min_max[10],c_min_max[11], 100)
		c3_X, c3_Y = np.meshgrid(c3_x,c3_y)
		c3_pos = np.empty(c3_X.shape + (2,))
		c3_pos[:, :, 0] = c3_X
		c3_pos[:, :, 1] = c3_Y
		c3_Z =G(c3_pos,mean3,inv_cov_matrix3)
		#plt.plot(x11,y11,'#DEB887',label='class1_predicted')
		#plt.plot(x22,y22,'#53868B',label='class2_predicted')
		#plt.plot(x33,y33,'#458B00',label='class3_predicted')
		plt.plot(x1,y1,'ro',label='class1_data')
		plt.plot(x2,y2,'bo',label='class2_data')
		plt.plot(x3,y3,'go',label='class3_data')
		plt.contour(c1_X,c1_Y,c1_Z, colors='red')
		plt.contour(c2_X,c2_Y,c2_Z, colors='blue')
		plt.contour(c3_X,c3_Y,c3_Z, colors='green')
		plt.legend()
		plt.show()    	

# this function is used to plot the data of any of the 2 classes with decions boundary
def plot_1_1(p_c1,p_c2,inv_cov_matrix1,inv_cov_matrix2,mean1,mean2,x1,x2,y1,y2,which_plot,c_min_max):
    
	x=np.zeros(2)
	x11=np.zeros(1000000)
	x22=np.zeros(1000000)
	y11=np.zeros(1000000)
	y22=np.zeros(1000000)
	minx=min(c_min_max[0],c_min_max[4],c_min_max[8])
	maxx=max(c_min_max[1],c_min_max[5],c_min_max[9])
	miny=min(c_min_max[2],c_min_max[6],c_min_max[10])
	maxy=max(c_min_max[3],c_min_max[7],c_min_max[11])
	x_intr=(maxx-minx)/250.00;
	y_intr=(maxy-miny)/250.00;
	l1=0
	l2=0
	i=minx-(0.05*(maxx-minx))
	while i < maxx+(0.05*(maxx-minx)):
                j=miny-(0.05*(maxy-miny));
		while j < maxy+(0.05*(maxy-miny)):
			x[0]=i
			x[1]=j
			clas=find_class.find_class_case1(p_c1,p_c2,inv_cov_matrix1,inv_cov_matrix2,mean1,mean2,x)
			if clas==0:
				x11[l1]=i
				y11[l1]=j
				l1=l1+1
			elif clas==1:	
				x22[l2]=i
				y22[l2]=j
				l2=l2+1

                        j=j+y_intr
                i=i+x_intr
    # plotting the data of class1 and class2 with decion boundary
	if which_plot==1:
		plt.plot(x11,y11,'#DEB887',label='class1_prediction')
		plt.plot(x22,y22,'#53868B',label='class2_prediction')
		plt.plot(x1,y1,'ro',label='class1_data')
		plt.plot(x2,y2,'bo',label='class2_data')
		plt.legend()
		plt.show()
    # plotting the data of class1 and class2 with contours of both the classes
	elif which_plot==-1:
		c1_pos=np.zeros((1000,1000,1000))
		c1_x = np.linspace(c_min_max[0],c_min_max[1], 100) 
		c1_y = np.linspace(c_min_max[2],c_min_max[3], 100)
		c1_X, c1_Y = np.meshgrid(c1_x,c1_y)    #
		c1_pos = np.empty(c1_X.shape + (2,))  
		c1_pos[:, :, 0] = c1_X
		c1_pos[:, :, 1] = c1_Y
		c1_Z =G(c1_pos,mean1,inv_cov_matrix1)   

		c2_pos=np.zeros((1000,1000,1000))
		c2_x = np.linspace(c_min_max[4],c_min_max[5], 100)
		c2_y = np.linspace(c_min_max[6],c_min_max[7], 100)
		c2_X, c2_Y = np.meshgrid(c2_x,c2_y)
		c2_pos = np.empty(c2_X.shape + (2,))
		c2_pos[:, :, 0] = c2_X
		c2_pos[:, :, 1] = c2_Y
		c2_Z =G(c2_pos,mean2,inv_cov_matrix2)


		plt.plot(x1,y1,'ro',label='class1_data')
		plt.plot(x2,y2,'bo',label='class2_data')
		plt.contour(c1_X,c1_Y,c1_Z, colors='red')
		plt.contour(c2_X,c2_Y,c2_Z, colors='blue')
		plt.legend()
		plt.show()

    # plotting the data of class2 and class3 with decion boundary
	elif which_plot==2:
		plt.plot(x11,y11,'#53868B',label='class2_prediction')
		plt.plot(x22,y22,'#458B00',label='class3_prediction')
		plt.plot(x1,y1,'bo',label='class2_data')
		plt.plot(x2,y2,'go',label='class3_data')
		plt.legend()
		plt.show()
     
    # plotting the data of class2 and class3 with contours of both the classes 
	elif which_plot==-2:
		c1_pos=np.zeros((1000,1000,1000))
		c1_x = np.linspace(c_min_max[4],c_min_max[5], 100) 
		c1_y = np.linspace(c_min_max[6],c_min_max[7], 100)
		c1_X, c1_Y = np.meshgrid(c1_x,c1_y)    #
		c1_pos = np.empty(c1_X.shape + (2,))  
		c1_pos[:, :, 0] = c1_X
		c1_pos[:, :, 1] = c1_Y
		c1_Z =G(c1_pos,mean1,inv_cov_matrix1)   

		c2_pos=np.zeros((1000,1000,1000))
		c2_x = np.linspace(c_min_max[8],c_min_max[9], 100)
		c2_y = np.linspace(c_min_max[10],c_min_max[11], 100)
		c2_X, c2_Y = np.meshgrid(c2_x,c2_y)
		c2_pos = np.empty(c2_X.shape + (2,))
		c2_pos[:, :, 0] = c2_X
		c2_pos[:, :, 1] = c2_Y
		c2_Z =G(c2_pos,mean2,inv_cov_matrix2)

		plt.plot(x1,y1,'bo',label='class2_data')
		plt.plot(x2,y2,'go',label='class3_data')
		plt.contour(c1_X,c1_Y,c1_Z, colors='blue')
		plt.contour(c2_X,c2_Y,c2_Z, colors='green')
		plt.legend()
		plt.show()    	
    # plotting the data of class1 and class3 with decion boundary
	elif which_plot==3:
		plt.plot(x11,y11,'#DEB887',label='class1_prediction')
		plt.plot(x22,y22,'#458B00',label='class3_prediction')
		plt.plot(x1,y1,'ro',label='class1_data')
		plt.plot(x2,y2,'go',label='class3_data')
		plt.legend()
		plt.show()	
    
    # plotting the data of class1 and class3 with contours of both the classes
	elif which_plot==-3:
		c1_pos=np.zeros((1000,1000,1000))
		c1_x = np.linspace(c_min_max[0],c_min_max[1], 100) 
		c1_y = np.linspace(c_min_max[2],c_min_max[3], 100)
		c1_X, c1_Y = np.meshgrid(c1_x,c1_y)    #
		c1_pos = np.empty(c1_X.shape + (2,))  
		c1_pos[:, :, 0] = c1_X
		c1_pos[:, :, 1] = c1_Y
		c1_Z =G(c1_pos,mean1,inv_cov_matrix1)   

		c2_pos=np.zeros((1000,1000,1000))
		c2_x = np.linspace(c_min_max[8],c_min_max[9], 100)
		c2_y = np.linspace(c_min_max[10],c_min_max[11], 100)
		c2_X, c2_Y = np.meshgrid(c2_x,c2_y)
		c2_pos = np.empty(c2_X.shape + (2,))
		c2_pos[:, :, 0] = c2_X
		c2_pos[:, :, 1] = c2_Y
		c2_Z =G(c2_pos,mean2,inv_cov_matrix2)

		plt.plot(x1,y1,'ro',label='class1_data')
		plt.plot(x2,y2,'go',label='class3_data')
		plt.contour(c1_X,c1_Y,c1_Z, colors='red')
		plt.contour(c2_X,c2_Y,c2_Z, colors='green')
		plt.legend()
		plt.show()    			

