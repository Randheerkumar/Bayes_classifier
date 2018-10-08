import numpy as np
import math

#this function find the class(out of all the three classes) for which the given data point has hghest prabalility or posteriar probility
def find_class_case(p_c1,p_c2,p_c3,inv_cov_matrix1,inv_cov_matrix2,inv_cov_matrix3,mean1,mean2,mean3,x) :

	g1=0.0;g2=0.0;g3=0.0;
	arr=np.zeros(3)
	_x1=np.zeros(2);
	_x2=np.zeros(2);
	_x3=np.zeros(2);
	_y1=np.zeros(2);
	_y2=np.zeros(2);
	_y3=np.zeros(2);

	_x1[0]=x[0]-mean1[0]
	_x1[1]=x[1]-mean1[1]

	_x2[0]=x[0]-mean2[0]
	_x2[1]=x[1]-mean2[1]

	_x3[0]=x[0]-mean3[0]
	_x3[1]=x[1]-mean3[1]

	_y1=_x1.transpose()
	_y2=_x2.transpose()
	_y3=_x3.transpose()

	_x1=np.matmul(_x1,inv_cov_matrix1)
	_x2=np.matmul(_x2,inv_cov_matrix2)
	_x3=np.matmul(_x3,inv_cov_matrix3)
	g1=(-0.5)*np.matmul(_x1,_y1)-(0.5)*math.log(np.linalg.det(np.linalg.inv(inv_cov_matrix1)))+math.log(p_c1)
	g2=(-0.5)*np.matmul(_x2,_y2)-(0.5)*math.log(np.linalg.det(np.linalg.inv(inv_cov_matrix2)))+math.log(p_c2)
	g3=(-0.5)*np.matmul(_x3,_y3)-(0.5)*math.log(np.linalg.det(np.linalg.inv(inv_cov_matrix3)))+math.log(p_c3)

	arr[0]=g1
	arr[1]=g2
	arr[2]=g3
	return(np.argmax(arr))


#this function find the class(out of any two of the three classes) for which the given data point has hghest prabalility or posteriar probility
def find_class_case1(p_c1,p_c2,inv_cov_matrix1,inv_cov_matrix2,mean1,mean2,x) :

	g1=0.0;g2=0.0;
	arr=np.zeros(2)
	_x1=np.zeros(2);
	_x2=np.zeros(2);
	_y1=np.zeros(2);
	_y2=np.zeros(2);

	_x1[0]=x[0]-mean1[0]
	_x1[1]=x[1]-mean1[1]

	_x2[0]=x[0]-mean2[0]
	_x2[1]=x[1]-mean2[1]


	_y1=_x1.transpose()
	_y2=_x2.transpose()

	_x1=np.matmul(_x1,inv_cov_matrix1)
	_x2=np.matmul(_x2,inv_cov_matrix2)
	g1=(-0.5)*np.matmul(_x1,_y1)-(0.5)*math.log(np.linalg.det(np.linalg.inv(inv_cov_matrix1)))+math.log(p_c1);
	g2=(-0.5)*np.matmul(_x2,_y2)-(0.5)*math.log(np.linalg.det(np.linalg.inv(inv_cov_matrix2)))+math.log(p_c2);

	arr[0]=g1
	arr[1]=g2
	return(np.argmax(arr))	