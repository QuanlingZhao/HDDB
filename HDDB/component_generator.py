from common import *
from tpcds_meta import *



def generate_component(table,str_max_dim,num,num_dim,d_per_level,level,str_base_dim,str_inc_dim,num_str_dim):
	print("Processing table: ",table)
	print("Generating encooders")
	base_encoder_obj = base_encoder(str_max_dim,num,num_dim,d_per_level,level)
	with open('./component/base_encoder_obj.pkl', 'wb') as f:
		pickle.dump(base_encoder_obj, f)
	print("reading_tbl_config")
	with open('../sample_tpcds_table/table_configs/'+ table +'_'+'meta_info.json', 'r') as file:
		tpcds_schema ,table_stats, encoding_plan = json.load(file)
	print('Encoding table')
	encode_tables([table],base_encoder_obj,tpc_ds_schema,table_stats,encoding_plan,num_str_dim)
	print('Done.')



































