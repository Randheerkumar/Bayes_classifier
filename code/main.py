'''
Pattern recognition cs669

lab 1 

Group10

'''
#importing Required libraries 
import numpy as np
import matplotlib.pyplot as plt                # for plotting
import math
import find_class
import find_cov_matrix
import test
import plot
import bayes_classifier


#variables for confusion matrix of all the 4 cases
confusion_matrix1=np.zeros((3,3))      
confusion_matrix2=np.zeros((3,3))
confusion_matrix3=np.zeros((3,3))
confusion_matrix4=np.zeros((3,3))

#for inverse covariances matrices of  all the classes
inv_cov_matrix1=np.zeros((2,2))
inv_cov_matrix2=np.zeros((2,2))
inv_cov_matrix3=np.zeros((2,2))

#for  covariances matrices of  all the classes
cov_matrix=np.zeros((2,2))
cov_matrix1=np.zeros((2,2))
cov_matrix2=np.zeros((2,2))
cov_matrix3=np.zeros((2,2))

#for mean vectors of  all the classes
mean1=np.zeros(2)
mean2=np.zeros(2)
mean3=np.zeros(2)
#this is for storing data point
x=np.zeros(2)

#for storing the data of class1
x1=np.zeros(10000)
y1=np.zeros(10000)

#for storing the data of class2
x2=np.zeros(10000)
y2=np.zeros(10000)

#for storing the data of class3
x3=np.zeros(10000)
y3=np.zeros(10000)

#for stpring the  maxm and minm x and y values of all the classes
c_min_max=np.zeros(12)

#taking the input file of all classess for traing data and testing data
'''
c1_train="class1.txt"
c2_train="class2.txt"
c3_train="class3.txt"
c1_test="class1_test.txt"
c2_test="class2_test.txt"
c3_test="class3_test.txt"
'''
c1_train=raw_input("enter 1st class training data file name\n")
c2_train=raw_input("enter 2nd class training data file name\n")
c3_train=raw_input("enter 3rd class training data file name\n")
c1_test=raw_input("enter 1st  class   testing data file name\n")
c2_test=raw_input("enter 2nd  class   testing data file name\n")
c3_test=raw_input("enter 3rd  class   testing data file name\n")

#array for stiring the number of training data in each aclass
p=np.zeros(3);
#here finding the covariances matrice of all the class by calling the function find_cov_matrix which is in find_cov_matrix.py file
cov_matrix1=find_cov_matrix.find_cov_matrix(mean1,x1,y1,c1_train,1,p,c_min_max)
cov_matrix2=find_cov_matrix.find_cov_matrix(mean2,x2,y2,c2_train,2,p,c_min_max)
cov_matrix3=find_cov_matrix.find_cov_matrix(mean3,x3,y3,c3_train,3,p,c_min_max)

#prabability all the classes
total=p[0]+p[1]+p[2]+0.000
p_c1=p[0]*1.000/total
p_c2=p[1]*1.000/total
p_c3=p[2]*1.000/total

'''
0: all three data with decision boundary
10: all 3 with cont

1: 1&2
-1: 1& 2 con
2:2,3
-2:2,3

3:1,3
-3:1,3

'''	

exit=1

while exit==1:

	choice=input("enter the case for which you want to evaluate\n1->case1\n2->case2\n3->case3\n4->case4\n")
	choice1=input("enter the choice\n0->class 1,2 & 3 data plot\n10->class 1,2 & 3 contour\n1->class 1 & 2 data plot\n-1->class 1 & 2 contour\n2->class 2 & 3 data plot \n-2->class2 & 3 contour\n3->class 1 & 3 data plot \n-3->class 1 & 3 contour \n")


	#case1:when covariance matrix is same diagonal and all the covariances are same

	if choice==1:
		if choice1==0:
			confusion_matrix1=bayes_classifier.case_1(c1_test,c2_test,c3_test,p_c1,p_c2,p_c3,cov_matrix1,cov_matrix2,cov_matrix3,mean1,mean2,mean3,x1,x2,x3,y1,y2,y3,choice1,c_min_max)
			print "confusion matrix is:\n"
			for i in range(3):
				for j in range(3):
					print confusion_matrix1[i][j]
		else:
			bayes_classifier.case_1(c1_test,c2_test,c3_test,p_c1,p_c2,p_c3,cov_matrix1,cov_matrix2,cov_matrix3,mean1,mean2,mean3,x1,x2,x3,y1,y2,y3,choice1,c_min_max)			

	#case2:when covariance matrix is same and full covariance matrix

	elif choice==2:
		if choice1==0:
			confusion_matrix2=bayes_classifier.case_2(c1_test,c2_test,c3_test,p_c1,p_c2,p_c3,cov_matrix1,cov_matrix2,cov_matrix3,mean1,mean2,mean3,x1,x2,x3,y1,y2,y3,choice1,c_min_max)
			#print "confusion matrux is:\n"
			for i in range(3):
				for j in range(3):
					print confusion_matrix2[i][j]
		else:
			bayes_classifier.case_2(c1_test,c2_test,c3_test,p_c1,p_c2,p_c3,cov_matrix1,cov_matrix2,cov_matrix3,mean1,mean2,mean3,x1,x2,x3,y1,y2,y3,choice1,c_min_max)

	#case3:when covariance matrix is diagonal and different for different class
	elif choice==3:
		if choice1 == 0:
			confusion_matrix3=bayes_classifier.case_3(c1_test,c2_test,c3_test,p_c1,p_c2,p_c3,cov_matrix1,cov_matrix2,cov_matrix3,mean1,mean2,mean3,x1,x2,x3,y1,y2,y3,choice1,c_min_max)
			#print "confusion matrix is:\n"
			for i in range(3):
				for j in range(3):
					print confusion_matrix3[i][j]
		else:
			bayes_classifier.case_3(c1_test,c2_test,c3_test,p_c1,p_c2,p_c3,cov_matrix1,cov_matrix2,cov_matrix3,mean1,mean2,mean3,x1,x2,x3,y1,y2,y3,choice1,c_min_max)
	#case4:when covariance matrix is different for different class

	elif choice==4:
		if choice1==0:
			confusion_matrix4=bayes_classifier.case_4(c1_test,c2_test,c3_test,p_c1,p_c2,p_c3,cov_matrix1,cov_matrix2,cov_matrix3,mean1,mean2,mean3,x1,x2,x3,y1,y2,y3,choice1,c_min_max)
			#print "confusion matrix is:\n"
			for i in range(3):
				for j in range(3):
					print confusion_matrix4[i][j]
		else :
		     bayes_classifier.case_4(c1_test,c2_test,c3_test,p_c1,p_c2,p_c3,cov_matrix1,cov_matrix2,cov_matrix3,mean1,mean2,mean3,x1,x2,x3,y1,y2,y3,choice1,c_min_max)			

	exit=input("enter 1 to cintinue or any number for exit \n")


