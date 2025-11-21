import random
from datetime import datetime
from tpcds_meta import *
import duckdb
import json
from filter_operator import *





def generate_random_filter_query(table,table_stats,filter_operator_possible_columns,possible_values):
	"""Generate a random TPC-DS-like single-table filter query."""
	predicate_type = random.choice(['string_col','numerical_col']) #['string_col','numerical_col']
	predicate_col = random.choice(filter_operator_possible_columns[table][predicate_type])

	if predicate_type == 'string_col':
		while True:
			value = random.choice(possible_values[table][predicate_col])
			if value != None:
				break
		predicate = f"{predicate_col} = {value}"

	if predicate_type == 'numerical_col':
		_max, _min = table_stats[table][predicate_col]
		value = random.randrange(int(_min),int(_max))
		relation = random.choice(['>','<'])
		predicate = f"{predicate_col} {relation} {value}"
	
	return f"SELECT row_id FROM {table}_with_rowid WHERE {predicate};"





def run_filter_operator(table,num_test,noise,num_str_dim):
	# prepare for query generator
	with open('../sample_tpcds_table/table_configs/'+ table +'_'+'meta_info.json', 'r') as file:
		tpcds_schema ,table_stats, encoding_plan = json.load(file)
	
	con = duckdb.connect()
	con.execute("PRAGMA threads=1")
	con.execute(TPCDS_DDL[table][0])
	con.execute(f"""COPY {table} FROM '{TPCDS_DDL[table][1]}' (DELIMITER '|');""")
	possible_values = {table:{}}
	for col_name in filter_operator_possible_columns[table]['string_col']:
		unique_values = con.execute(f"""SELECT DISTINCT {col_name} FROM {table}""").fetchall()
		unique_values = [x[0] for x in unique_values]
		possible_values[table][col_name] = unique_values

	con.execute(f"""CREATE OR REPLACE TABLE {table}_with_rowid AS SELECT row_number() OVER () AS row_id, * FROM {table};""")

	# generate ground truth query results
	queries = []
	real_results = []

	for i in range(num_test):
		queries.append(generate_random_filter_query(table,table_stats,filter_operator_possible_columns,possible_values))


	for i in range(num_test):
		result = con.execute(queries[i]).fetchall()
		result = [x[0] for x in result]
		real_results.append(result)


	# generate hddb results
	hddb_results = []

	with open('./component/base_encoder_obj.pkl', 'rb') as file:
		encoder = pickle.load(file)
	table_encoding_path = './encoded_tables/'+table+'.npz'
	filtering_operator = hddb_fliter_operator(table,table_encoding_path,tpcds_schema,table_stats,encoding_plan,encoder,noise,num_str_dim)


	for i in range(num_test):
		results = filtering_operator.process_predicate(queries[i])
		hddb_results.append(results)


	# evaluation
	correct = 0
	for i in  range(num_test):
		if set(hddb_results[i]) == set(real_results[i]):
			correct+=1
		else:
			print(len(hddb_results[i]),'--------',len(real_results[i]),'--------',queries[i])
	print('===================')
	print("Noise: ",noise)
	print("Num of test: ",num_test)
	print("Final Accuracy: ", correct / num_test)
	print('===================')
	return correct / num_test






































