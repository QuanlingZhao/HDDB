from component_generator import *
from run_filter_exp import *
from run_projection_exp import *
import matplotlib.pyplot as plt


'''
#110K
table = "catalog_sales"
str_max_dim = 10002
num = 102
num_dim = 2004
d_per_level = 501
level = 4
str_base_dim = 999
str_inc_dim = 201
num_str_dim = 2502
'''

l = 11


table = "catalog_sales"
str_max_dim = 10002
num = 102
num_dim = (int((501 / 11) * l) + (3-(int((501 / 11) * l) % 3))) * 4
d_per_level = int((501 / 11) * l) + (3-(int((501 / 11) * l) % 3)) 
level = 4
str_base_dim = int((999 / 11) * l) + (3-(int((999 / 11) * l) % 3))
str_inc_dim = int((201 / 11) * l) + (3-(int((201 / 11) * l) % 3))
num_str_dim = int((2502 / 11) * l) + (3-(int((2502 / 11) * l) % 3))




num_test = 100
noise = [0.03 * i for i in range(11)]


if __name__ == '__main__':

	flt_rlt = []
	poj_rlt = []

	
	generate_component(table,str_max_dim,num,num_dim,d_per_level,level,str_base_dim,str_inc_dim,num_str_dim)
	for n in noise:
		print('################################################ noise level =',n)
		flt_rlt.append(run_filter_operator(table,num_test,n,num_str_dim))
		poj_rlt.append(run_projection_operator(table,num_test,n,num_str_dim))

	print("######################Report######################")
	print('flt_rlt: ',flt_rlt)
	print('poj_rlt: ',poj_rlt)


	plt.plot(noise, flt_rlt, label='Filter Operator')
	plt.plot(noise, poj_rlt, label='Projection Operator')
	plt.legend(loc='lower left')
	plt.show()

















































