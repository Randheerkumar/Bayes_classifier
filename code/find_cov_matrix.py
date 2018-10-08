#importing various library
import numpy as np  
import math


def find_cov_matrix(mean1,x1,y1,file,which_class,p,c_min_max) :
    
    #definig varibles for find mean and covariance matrix of the given class
	x1_sum=0.0000000;y1_sum=0.000000;xy_sum=0.0000000;   
	x1_square_sum=0.000000;y1_square_sum=0.00000000;
	cov_matrix=np.zeros((2,2))
	minx=10000000
	miny=10000000
	maxx=-10000000
	maxy=-10000000
	count=0;                                            # counting the number of data in this class
	i=0;
	file1=open(file,"r")                                # taking input from file
	for line in file1:
		a=line.split()
		x1[i]= float(a[0])
		y1[i]= float(a[1])

		x1_sum =x1_sum + x1[i]	
		y1_sum = y1_sum + y1[i]
		xy_sum += x1[i]*y1[i]
		x1_square_sum += x1[i]*x1[i]
		y1_square_sum += y1[i]*y1[i]
		if minx>x1[i]:
			minx=x1[i]
		if miny>y1[i]:
   		    miny=y1[i]	
		if maxx<x1[i]:
			maxx=x1[i]
		if maxy<y1[i]:
			maxy=y1[i]	

		i+=1
   
	#finding the sample mean and covariance matrix
	if which_class==1:
		p[0]=i
		c_min_max[0]=minx
		c_min_max[1]=maxx
		c_min_max[2]=miny
		c_min_max[3]=maxy
	elif which_class==2:
		p[1]=i
		c_min_max[4]=minx
		c_min_max[5]=maxx
		c_min_max[6]=miny
		c_min_max[7]=maxy		
	elif which_class==3:
		p[2]=i
		c_min_max[8]=minx
		c_min_max[9]=maxx
		c_min_max[10]=miny
		c_min_max[11]=maxy		

	count=i
	mean_x1=x1_sum/count
	mean_y1=y1_sum/count
	cov_xx=x1_square_sum/count - mean_x1*mean_x1
	cov_yy=y1_square_sum/count - mean_y1*mean_y1
	cov_xy=(xy_sum)/count - mean_x1*mean_y1
	cov_yx=cov_xy

	#forming the covariannce matrix for this class using all the covariances
	cov_matrix[0][0]=cov_xx
	cov_matrix[0][1]=cov_xy
	cov_matrix[1][0]=cov_yx
	cov_matrix[1][1]=cov_yy

	#mean vector of this class 
	mean1[0]=mean_x1
	mean1[1]=mean_y1
	return(cov_matrix)   
