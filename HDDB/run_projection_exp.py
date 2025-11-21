import random
from datetime import datetime
from tpcds_meta import *
import duckdb
import json
from projection_operator import *






def generate_random_projection_query(table,projection_operator_possible_columns):
	col = random.choice(projection_operator_possible_columns)
	return f"SELECT {col} FROM {table};"





def run_projection_operator(table,num_test,noise,num_str_dim):
	# prepare for query generator
	with open('../sample_tpcds_table/table_configs/'+ table +'_'+'meta_info.json', 'r') as file:
		tpcds_schema ,table_stats, encoding_plan = json.load(file)
	
	con = duckdb.connect()
	con.execute("PRAGMA threads=1")
	con.execute(TPCDS_DDL[table][0])
	con.execute(f"""COPY {table} FROM '{TPCDS_DDL[table][1]}' (DELIMITER '|');""")
	for col_name in filter_operator_possible_columns[table]['string_col']:
		unique_values = con.execute(f"""SELECT DISTINCT {col_name} FROM {table}""").fetchall()
		unique_values = [x[0] for x in unique_values]
	con.execute(f"""CREATE OR REPLACE TABLE {table}_with_rowid AS SELECT row_number() OVER () AS row_id, * FROM {table};""")

	# generate ground truth query results
	queries = []
	real_results = []

	for i in range(num_test):
		queries.append(generate_random_projection_query(table,list(tpcds_schema['catalog_sales'].keys())))



	for i in range(num_test):
		result = con.execute(queries[i]).fetchall()
		result = [float(x[0]) if x[0] not in [None] else None for x in result]
		real_results.append(result)



	# generate hddb results
	hddb_results = []

	with open('./component/base_encoder_obj.pkl', 'rb') as file:
		encoder = pickle.load(file)
	table_encoding_path = './encoded_tables/'+table+'.npz'
	projection_operator = hddb_projection_operator(table,table_encoding_path,tpcds_schema,table_stats,encoding_plan,encoder,noise,num_str_dim)


	for i in range(num_test):
		try:
			results = projection_operator.process_projection(queries[i])
			hddb_results.append(results)
		except:
			hddb_results.append([])


	# evaluation
	correct = 0
	for i in  range(num_test):
		if hddb_results[i] == real_results[i]:
			correct+=1
		else:
			print('Error')
			for j in range(min([len(real_results[i]),len(hddb_results[i])])):
				if real_results[i][j]!=hddb_results[i][j]:
					print(real_results[i][j],hddb_results[i][j])

	print('===================')
	print("Noise: ",noise)
	print("Num of test: ",num_test)
	print("Final Accuracy: ", correct / num_test)
	print('===================')
	return correct / num_test



































