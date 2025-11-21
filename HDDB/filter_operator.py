import pickle
import cupy as np
from common import *
from mlc_common import *





class hddb_fliter_operator():
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



	def process_predicate(self,query):
		predicate = query.split('WHERE')[1][1:-1]
		column = predicate.split(' ')[0]
		predicate_type = self.schema[self.table][column]
		working_dim = self.plan[self.table][column][0]
		strat_dim = self.plan[self.table][column][1][0]
		end_dim = self.plan[self.table][column][1][1]
		
		#print(predicate)
		#print(column)
		#print(predicate_type)
		#print(working_dim,strat_dim,end_dim)
		
		if predicate_type == 'string':
			value = predicate.split(' ')[2]
			encoded_value = binary_to_LC((np.array([self.encoder.string_encoder.encode(value,working_dim)])>0).astype('int'))
			selected_table = self.table_data[:,int(strat_dim/3):int(end_dim/3)]
			
			#print(encoded_value)
			#print(selected_table)
			#print(encoded_value.shape)
			#print(selected_table.shape)
			#exit()

			sims = self.mlc_func.sims(selected_table,encoded_value)

			threshold = (working_dim / 3) * 0.8
			retrived_indices = (np.where(sims > threshold)[0] + 1).tolist()

			return retrived_indices
		
		
		if predicate_type == 'numerical':
			_max, _min = self.stats[self.table][column]
			relation = predicate.split(' ')[1]
			val_enc = binary_to_LC(np.array([(self.encoder.number_encoder.encode(float(predicate.split(' ')[2]),_min,_max)>0).astype('int')]).astype('int'))[0]
			selected_table = self.table_data[:,int(strat_dim/3):int(end_dim/3)]

			selected_val = selected_table[:,:self.encoder.number_encoder.D / 3]
			selected_str = selected_table[:,self.encoder.number_encoder.D / 3:]

			#print(selected_table.shape)
			#print(selected_val.shape)
			#print(selected_str.shape)
			#print(val_enc.shape)

			q_encs = np.array_split(val_enc, self.encoder.number_encoder.level)
			b_encs = np.array_split(selected_val, self.encoder.number_encoder.level,axis=1)

			#for c in q_encs:
			#	print(c.shape)
			#for c in b_encs:
			#	print(c.shape)

			#print(cross_hamming_general(np.expand_dims(q_encs[0],axis=0),self.mlc_func.range_hvs))
			#print(cross_hamming_general(b_encs[0],self.mlc_func.range_hvs))



			q_indicator = []
			b_indicator = []
			for i in range(self.encoder.number_encoder.level):
				q_indicator.append(np.argmin(cross_hamming_general(np.expand_dims(q_encs[i],axis=0),self.mlc_func.range_hvs)))
				b_indicator.append(np.argmin(cross_hamming_general(b_encs[i],self.mlc_func.range_hvs),axis=1))
			q_indicator = np.array(q_indicator)
			b_indicator = np.array(b_indicator).T



			pseudo = np.array([self.encoder.number_encoder.num**i for i in range(self.encoder.number_encoder.level)])[::-1]

			#print(pseudo)

			q_val = np.dot(q_indicator,pseudo)
			b_val = np.multiply(np.tile(pseudo, (len(selected_val), 1)),b_indicator).sum(axis = 1)
			
			nan = (self.mlc_func.sims(selected_val,self.mlc_func.nan) / selected_val.shape[1]) > (0.8)

			if relation == ">":
				return (np.where((b_val > q_val) & (nan != True))[0] + 1).tolist()
			if relation == "<":
				return (np.where((b_val < q_val) & (nan != True))[0] + 1).tolist()








































