import numpy as np
import math
import find_class

#this function tests the testing data of all the classes and make the corresponding confusion matrix

def test_1(c1_test,c2_test,c3_test,p_c1,p_c2,p_c3,inv_cov_matrix1,inv_cov_matrix2,inv_cov_matrix3,mean1,mean2,mean3) :

	confusion_matrix=np.zeros((3,3))
	x=np.zeros(2)
	file1=open(c1_test,"r")
	for line in file1:
		a=line.split()
		x[0]= float(a[0])
		x[1]=float(a[1])
		clas=find_class.find_class_case(p_c1,p_c2,p_c3,inv_cov_matrix1,inv_cov_matrix2,inv_cov_matrix3,mean1,mean2,mean3,x)
		confusion_matrix[0][clas]+=1


	file2=open(c2_test,"r")
	for line in file2:
		a=line.split()
		x[0]= float(a[0])
		x[1]=float(a[1])
		clas=find_class.find_class_case(p_c1,p_c2,p_c3,inv_cov_matrix1,inv_cov_matrix2,inv_cov_matrix3,mean1,mean2,mean3,x)
		confusion_matrix[1][clas]+=1	


	file3=open(c3_test,"r")
	for line in file3:
		a=line.split()
		x[0]= float(a[0])
		x[1]=float(a[1])
		clas=find_class.find_class_case(p_c1,p_c2,p_c3,inv_cov_matrix1,inv_cov_matrix2,inv_cov_matrix3,mean1,mean2,mean3,x)
		confusion_matrix[2][clas]+=1


	return(confusion_matrix)	


