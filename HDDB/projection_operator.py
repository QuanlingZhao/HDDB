import pickle
import cupy as np
from component_generator import *
from common import *
from mlc_common import *


class hddb_projection_operator():
	def __init__(self,table,table_encoding_path,tpcds_schema,table_stats,encoding_plan,encoder,noise,num_str_dim):
		self.table = table
		self.path = table_encoding_path
		self.schema = tpcds_schema
		self.stats = table_stats
		self.plan = encoding_plan
		self.encoder = encoder
		self.noise = noise
		self.num_str_dim = num_str_dim

		self.packed = np.load(table_encoding_path, allow_pickle=False)
		self.shape = (self.packed['shape'][0].get(),self.packed['shape'][1].get())
		self.table_data = np.array(self.packed['data'])[:self.shape[0]*self.shape[1]].astype(np.uint8)

		del self.packed

		if noise != 0:
			num_corrupt = int(self.table_data.shape[0]*self.table_data.shape[1]* (self.noise / 2))
			row_idx = np.random.choice(self.table_data.shape[0],num_corrupt,replace=True)
			column_idx = np.random.choice(self.table_data.shape[1],num_corrupt,replace=True)
			self.table_data[row_idx,column_idx] = self.table_data[row_idx,column_idx] - 1

			num_corrupt = int(self.table_data.shape[0]*self.table_data.shape[1]* (self.noise / 2))
			row_idx = np.random.choice(self.table_data.shape[0],num_corrupt,replace=True)
			column_idx = np.random.choice(self.table_data.shape[1],num_corrupt,replace=True)
			self.table_data[row_idx,column_idx] = self.table_data[row_idx,column_idx] + 1

			self.table_data[self.table_data>7] = 7
			self.table_data[self.table_data<0] = 0

		self.mlc_func = mlc_common(encoder)



	def process_projection(self,query):
		column = query.split(' ')[1]
		col_type = self.schema[self.table][column]
		working_dim = self.plan[self.table][column][0]
		strat_dim = self.plan[self.table][column][1][0]
		end_dim = self.plan[self.table][column][1][1]


		if col_type == 'numerical':
			working_dim = self.num_str_dim
			strat_dim = end_dim-working_dim
		selected_table = self.table_data[:,int(strat_dim/3):int(end_dim/3)]



		decoded_column = []
		for i in range(len(selected_table)):
			cell = self.mlc_func.string_decode(selected_table[i])
			if cell == '':
				decoded_column.append(None)
			else:
				decoded_column.append(float(cell))
		return decoded_column








































