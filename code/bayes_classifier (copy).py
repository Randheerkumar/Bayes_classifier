import numpy as np
import test
import plot
import math

# this case calculate the confusion matrix ,plot the data with decion boundary depending for case1 when covariance matrix  is same and diagonal and all the covariances are same
def case_1(c1_test,c2_test,c3_test,p_c1,p_c2,p_c3,cov_matrix1,cov_matrix2,cov_matrix3,mean1,mean2,mean3,x1,x2,x3,y1,y2,y3,which_plot,c_min_max):
    
	cov_matrix=np.zeros((2,2))
	covv_matrix=np.zeros((2,2))

	inv_cov_matrix=np.zeros((2,2))
	inv_cov_matrix1=np.zeros((2,2))
	inv_cov_matrix2=np.zeros((2,2))
	inv_cov_matrix3=np.zeros((2,2))
	confusion_matrix=np.zeros((3,3))
	for i in range(2):
		for j in range(2):
			if (i+j) %2==0:
				covv_matrix[i][j]=(cov_matrix1[i][j]+cov_matrix2[i][j]+cov_matrix3[i][j])/3;

			else:
			    cov_matrix[i][j]=0;	
	cov_matrix[0][0]=(covv_matrix[0][0]+covv_matrix[1][1])/2.0000;
	cov_matrix[1][1]=(covv_matrix[0][0]+covv_matrix[1][1])/2.0000;
	inv_cov_matrix=np.linalg.inv(cov_matrix)
	inv_cov_matrix1=inv_cov_matrix2=inv_cov_matrix3=inv_cov_matrix
	#confusion_matrix=test.test_1(c1_test,c2_test,c3_test,inv_cov_matrix1,inv_cov_matrix2,inv_cov_matrix3,mean1,mean2,mean3)
	if which_plot==0:
		confusion_matrix=test.test_1(c1_test,c2_test,c3_test,p_c1,p_c2,p_c3,inv_cov_matrix1,inv_cov_matrix2,inv_cov_matrix3,mean1,mean2,mean3)
		plot.plot_1(p_c1,p_c2,p_c3,inv_cov_matrix1,inv_cov_matrix2,inv_cov_matrix3,mean1,mean2,mean3,x1,x2,x3,y1,y2,y3,0,c_min_max)
		return(confusion_matrix)        
	elif which_plot==10:
		plot.plot_1(p_c1,p_c2,p_c3,inv_cov_matrix1,inv_cov_matrix2,inv_cov_matrix3,mean1,mean2,mean3,x1,x2,x3,y1,y2,y3,10,c_min_max)	
	elif which_plot==-1:
		plot.plot_1_1(p_c1,p_c2,inv_cov_matrix1,inv_cov_matrix2,mean1,mean2,x1,x2,y1,y2,-1,c_min_max)
	elif which_plot==1:
		plot.plot_1_1(p_c1,p_c2,inv_cov_matrix1,inv_cov_matrix2,mean1,mean2,x1,x2,y1,y2,1,c_min_max)	
	elif which_plot==-2:
	    plot.plot_1_1(p_c2,p_c3,inv_cov_matrix2,inv_cov_matrix3,mean2,mean3,x2,x3,y2,y3,-2,c_min_max)
	elif which_plot==2:
	    plot.plot_1_1(p_c2,p_c3,inv_cov_matrix2,inv_cov_matrix3,mean2,mean3,x2,x3,y2,y3,2,c_min_max)    
	elif which_plot==-3:
	    plot.plot_1_1(p_c1,p_c3,inv_cov_matrix1,inv_cov_matrix3,mean1,mean3,x1,x3,y1,y3,-3,c_min_max) 
	elif which_plot==3:
	    plot.plot_1_1(p_c1,p_c3,inv_cov_matrix1,inv_cov_matrix3,mean1,mean3,x1,x3,y1,y3,3,c_min_max)     

	#return(confusion_matrix)

#when the covariance matrices of all the classes are same and arbitrary
def case_2(c1_test,c2_test,c3_test,p_c1,p_c2,p_c3,cov_matrix1,cov_matrix2,cov_matrix3,mean1,mean2,mean3,x1,x2,x3,y1,y2,y3,which_plot,c_min_max):
    
	cov_matrix=np.zeros((2,2))
	inv_cov_matrix=np.zeros((2,2))
	inv_cov_matrix1=np.zeros((2,2))
	inv_cov_matrix2=np.zeros((2,2))
	inv_cov_matrix3=np.zeros((2,2))
	confusion_matrix=np.zeros((3,3))
	for i in range(2):
		for j in range(2):
			cov_matrix[i][j]=(cov_matrix1[i][j]+cov_matrix2[i][j]+cov_matrix3[i][j])/3;


	inv_cov_matrix=np.linalg.inv(cov_matrix)
	inv_cov_matrix1=inv_cov_matrix2=inv_cov_matrix3=inv_cov_matrix
	#confusion_matrix=test.test_1(c1_test,c2_test,c3_test,p_c1,p_c2,p_c3,inv_cov_matrix1,inv_cov_matrix2,inv_cov_matrix3,mean1,mean2,mean3)
	if which_plot==0:
		confusion_matrix=test.test_1(c1_test,c2_test,c3_test,p_c1,p_c2,p_c3,inv_cov_matrix1,inv_cov_matrix2,inv_cov_matrix3,mean1,mean2,mean3)
		plot.plot_1(p_c1,p_c2,p_c3,inv_cov_matrix1,inv_cov_matrix2,inv_cov_matrix3,mean1,mean2,mean3,x1,x2,x3,y1,y2,y3,0,c_min_max)
		return(confusion_matrix)
	elif which_plot==10:
		plot.plot_1(p_c1,p_c2,p_c3,inv_cov_matrix1,inv_cov_matrix2,inv_cov_matrix3,mean1,mean2,mean3,x1,x2,x3,y1,y2,y3,10,c_min_max)	
	elif which_plot==-1:
		plot.plot_1_1(p_c1,p_c2,inv_cov_matrix1,inv_cov_matrix2,mean1,mean2,x1,x2,y1,y2,-1,c_min_max)
	elif which_plot==1:
		plot.plot_1_1(p_c1,p_c2,inv_cov_matrix1,inv_cov_matrix2,mean1,mean2,x1,x2,y1,y2,1,c_min_max)	
	elif which_plot==-2:
	    plot.plot_1_1(p_c2,p_c3,inv_cov_matrix2,inv_cov_matrix3,mean2,mean3,x2,x3,y2,y3,-2,c_min_max)
	elif which_plot==2:
	    plot.plot_1_1(p_c2,p_c3,inv_cov_matrix2,inv_cov_matrix3,mean2,mean3,x2,x3,y2,y3,2,c_min_max)    
	elif which_plot==-3:
	    plot.plot_1_1(p_c1,p_c3,inv_cov_matrix1,inv_cov_matrix3,mean1,mean3,x1,x3,y1,y3,-3,c_min_max) 
	elif which_plot==3:
	    plot.plot_1_1(p_c1,p_c3,inv_cov_matrix1,inv_cov_matrix3,mean1,mean3,x1,x3,y1,y3,3,c_min_max)        

	#return(confusion_matrix)	

#when  covariance matrices of all classes are different and diagonal
def case_3(c1_test,c2_test,c3_test,p_c1,p_c2,p_c3,cov_matrix1,cov_matrix2,cov_matrix3,mean1,mean2,mean3,x1,x2,x3,y1,y2,y3,which_plot,c_min_max):
    
	cov_matrix=np.zeros((2,2))
	cov_matrix1_1=np.zeros((2,2))
	cov_matrix2_2=np.zeros((2,2))
	cov_matrix3_3=np.zeros((2,2))
	cov_matrix1_1[0][0]=cov_matrix1[0][0]
	cov_matrix2_2[0][0]=cov_matrix2[0][0]
	cov_matrix3_3[0][0]=cov_matrix3[0][0]
	cov_matrix1_1[1][1]=cov_matrix1[1][1]
	cov_matrix2_2[1][1]=cov_matrix2[1][1]
	cov_matrix3_3[1][1]=cov_matrix3[1][1]

	inv_cov_matrix1=np.zeros((2,2))
	inv_cov_matrix2=np.zeros((2,2))
	inv_cov_matrix3=np.zeros((2,2))
	confusion_matrix=np.zeros((3,3))
	'''
	for i in range(2):
		for j in range(2):
			if(i+j)%2!=0:
				cov_matrix1_1[i][j]=0;
				cov_matrix2_2[i][j]=0;
				cov_matrix3_3[i][j]=0;
    '''

	inv_cov_matrix1=np.linalg.inv(cov_matrix1_1)
	inv_cov_matrix2=np.linalg.inv(cov_matrix2_2)
	inv_cov_matrix3=np.linalg.inv(cov_matrix3_3)
	#confusion_matrix=test.test_1(c1_test,c2_test,c3_test,inv_cov_matrix1,inv_cov_matrix2,inv_cov_matrix3,mean1,mean2,mean3)
	if which_plot==0:
		confusion_matrix=test.test_1(c1_test,c2_test,c3_test,p_c1,p_c2,p_c3,inv_cov_matrix1,inv_cov_matrix2,inv_cov_matrix3,mean1,mean2,mean3)
		plot.plot_1(p_c1,p_c2,p_c3,inv_cov_matrix1,inv_cov_matrix2,inv_cov_matrix3,mean1,mean2,mean3,x1,x2,x3,y1,y2,y3,0,c_min_max)
		return(confusion_matrix)		
	elif which_plot==10:
		plot.plot_1(p_c1,p_c2,p_c3,inv_cov_matrix1,inv_cov_matrix2,inv_cov_matrix3,mean1,mean2,mean3,x1,x2,x3,y1,y2,y3,10,c_min_max)	
	elif which_plot==-1:
		plot.plot_1_1(p_c1,p_c2,inv_cov_matrix1,inv_cov_matrix2,mean1,mean2,x1,x2,y1,y2,-1,c_min_max)
	elif which_plot==1:
		plot.plot_1_1(p_c1,p_c2,inv_cov_matrix1,inv_cov_matrix2,mean1,mean2,x1,x2,y1,y2,1,c_min_max)	
	elif which_plot==-2:
	    plot.plot_1_1(p_c2,p_c3,inv_cov_matrix2,inv_cov_matrix3,mean2,mean3,x2,x3,y2,y3,-2,c_min_max)
	elif which_plot==2:
	    plot.plot_1_1(p_c2,p_c3,inv_cov_matrix2,inv_cov_matrix3,mean2,mean3,x2,x3,y2,y3,2,c_min_max)    
	elif which_plot==-3:
	    plot.plot_1_1(p_c1,p_c3,inv_cov_matrix1,inv_cov_matrix3,mean1,mean3,x1,x3,y1,y3,-3,c_min_max) 
	elif which_plot==3:
	    plot.plot_1_1(p_c1,p_c3,inv_cov_matrix1,inv_cov_matrix3,mean1,mean3,x1,x3,y1,y3,3,c_min_max)        
	#return(confusion_matrix)		
	
#when covariance matrices of all class are different and arbitrary 
def case_4(c1_test,c2_test,c3_test,p_c1,p_c2,p_c3,cov_matrix1,cov_matrix2,cov_matrix3,mean1,mean2,mean3,x1,x2,x3,y1,y2,y3,which_plot,c_min_max):
    
	cov_matrix=np.zeros((2,2))
	inv_cov_matrix1=np.zeros((2,2))
	inv_cov_matrix2=np.zeros((2,2))
	inv_cov_matrix3=np.zeros((2,2))
	confusion_matrix=np.zeros((3,3))

	inv_cov_matrix1=np.linalg.inv(cov_matrix1)
	inv_cov_matrix2=np.linalg.inv(cov_matrix2)
	inv_cov_matrix3=np.linalg.inv(cov_matrix3)
	#confusion_matrix=test.test_1(c1_test,c2_test,c3_test,inv_cov_matrix1,inv_cov_matrix2,inv_cov_matrix3,mean1,mean2,mean3)
	if which_plot==0:
		confusion_matrix=test.test_1(c1_test,c2_test,c3_test,p_c1,p_c2,p_c3,inv_cov_matrix1,inv_cov_matrix2,inv_cov_matrix3,mean1,mean2,mean3)
		plot.plot_1(p_c1,p_c2,p_c3,inv_cov_matrix1,inv_cov_matrix2,inv_cov_matrix3,mean1,mean2,mean3,x1,x2,x3,y1,y2,y3,0,c_min_max)
		return(confusion_matrix)		
	elif which_plot==10:
		plot.plot_1(p_c1,p_c2,p_c3,inv_cov_matrix1,inv_cov_matrix2,inv_cov_matrix3,mean1,mean2,mean3,x1,x2,x3,y1,y2,y3,10,c_min_max)	
	elif which_plot==-1:
		plot.plot_1_1(p_c1,p_c2,inv_cov_matrix1,inv_cov_matrix2,mean1,mean2,x1,x2,y1,y2,-1,c_min_max)
	elif which_plot==1:
		plot.plot_1_1(p_c1,p_c2,inv_cov_matrix1,inv_cov_matrix2,mean1,mean2,x1,x2,y1,y2,1,c_min_max)	
	elif which_plot==-2:
	    plot.plot_1_1(p_c2,p_c3,inv_cov_matrix2,inv_cov_matrix3,mean2,mean3,x2,x3,y2,y3,-2,c_min_max)
	elif which_plot==2:
	    plot.plot_1_1(p_c2,p_c3,inv_cov_matrix2,inv_cov_matrix3,mean2,mean3,x2,x3,y2,y3,2,c_min_max)    
	elif which_plot==-3:
	    plot.plot_1_1(p_c1,p_c3,inv_cov_matrix1,inv_cov_matrix3,mean1,mean3,x1,x3,y1,y3,-3,c_min_max) 
	elif which_plot==3:
	    plot.plot_1_1(p_c1,p_c3,inv_cov_matrix1,inv_cov_matrix3,mean1,mean3,x1,x3,y1,y3,3,c_min_max)        
 
 	#return(confusion_matrix)		
		
	
		
	

