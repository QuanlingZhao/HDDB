import random
import string
import cupy as np
from cupy import dot
from cupy.linalg import norm
import matplotlib.pyplot as plt
import copy
import time
from scipy import stats
import csv
import pickle
import math
import numpy
import pandas as pd
import json



class string_encoder_decoder:
    def __init__(self,dim):
        self.dim = dim
        #self.ch = [chr(c) for c in range(0x20, 0x7F) if c != 0x7C]
        self.ch = ['-','.','1','2','3','4','5','6','7','8','9','0']
        self.ch = numpy.array(self.ch)
        self.alp = np.sign(self.gram_schmidt(np.random.normal(0,1,(len(self.ch)+1+150,dim)).T).T)
        #self.T = self.alp[len(self.ch)]
        self.char_to_index = {char: idx for idx, char in enumerate(self.ch)}
        self.I_unrolled = self.alp[len(self.ch)+1:]
        assert len(self.I_unrolled) == 150
        self.even_pad = np.array([len(self.ch)]*3)
        self.odd_pad = np.array([len(self.ch)]*2)


        
    def gram_schmidt(self,A):
        #n = A.shape[1]
        #for j in range(n):
        #    for k in range(j):
        #        A[:, j] -= np.dot(A[:, k], A[:, j]) * A[:, k]
        #    A[:, j] = A[:, j] / np.linalg.norm(A[:, j])
        return A
    
    def check_codebook_sim(self):
        sims = (self.alp @ self.alp.T) / self.dim
        plt.imshow(sims, cmap='hot', interpolation='nearest')
        
    def encode(self,s,working_dim):
        assert(type(s) == str) and (len(s)<=140)
        assert(working_dim <= self.dim)
        working_alp = self.alp[:,:working_dim]
        working_I_unrolled = self.I_unrolled[:,:working_dim]
        
        s = ''.join([c if c in self.ch else "" for c in s])
        
        length = len(s)
        ids = np.array([self.char_to_index[char] for char in s])
        if length%2 == 0:
            ids = np.hstack((ids,self.even_pad))
        else:
            ids = np.hstack((ids,self.odd_pad))
        ids = ids.astype('int16')
        encoded = np.sign(np.multiply(working_alp[ids],working_I_unrolled[:len(ids)]).sum(axis=0))      
        return encoded

    def decode(self,e):
        working_dim = len(e)
        assert(working_dim <= self.dim)
        working_ch = self.ch
        working_alp = self.alp[:,:working_dim]
        working_I_unrolled = self.I_unrolled[:,:working_dim]
        ids = np.argmax(np.multiply(np.repeat(np.expand_dims(e, axis=0), 150, axis=0),working_I_unrolled) @ working_alp[:len(working_ch)+1].T, axis=1)
        try:
            cut_off = np.where(ids == len(self.ch))[0][0]
            ids = ids[:cut_off]
        except:
            pass
        string = ''.join(working_ch[ids.get()])
        return string

    

class number_encoder_decoder():
    def __init__(self,num,D, d_per_level, level):
        assert d_per_level * level == D
        self.D = D
        self.d_per_level = d_per_level
        self.level = level
        self.num = num
        self.init_hvs()
        
    def initial_hv_processing(self,A):
        return A

    def init_hvs(self):
        self.range_hvs = np.sign(self.initial_hv_processing(np.random.normal(0,1,(self.num+1,self.d_per_level)).T).T)
        self.nan = np.sign(np.random.normal(0,1,(self.D)))
    
    def encode(self,number,_min,_max):
        if number == "":
            return self.nan
        assert _min < _max
        assert number<=_max and number>=_min
        _range = _max - _min
        interval = _range/self.num
        rs = []
        excess = (number - _min)
        for i in range(self.level):
            lvl = math.floor(excess/interval)
            rs.append(self.range_hvs[lvl].copy())
            
            excess = excess - lvl * interval
            _min = 0
            _max = interval
            _range = _max - _min
            interval = _range/self.num
        out = np.hstack(rs)
        assert len(out) == self.D
        return out
    
    def check_nan(self,number_enc):
        return ((number_enc == self.nan).sum() > (0.8)*(self.D))
        
    
    def decode(self,number_enc,_min,_max):
        assert _min < _max
        if self.check_nan(number_enc):
            return ""
        
        _range = _max - _min
        interval = _range/self.num
        reconstruct = _min
        encs = np.array_split(number_enc, self.level)

        for enc in encs:
            reconstruct += np.argmax(enc @ self.range_hvs.T) * interval
            interval = interval/self.num

        return reconstruct


    def recon(self,range_indicies,_min,_max):
        _range = _max - _min
        interval = _range/self.num
        reconstruct = _min
        for i in range_indicies:
            reconstruct += i * interval
            interval = interval/self.num
        return reconstruct

        
    def compare(self,a,b):
        a = self.decode(a,-100,100)
        b = self.decode(b,-100,100)
        if a>b:
            return 1
        if a<b:
            return 2


            
class base_encoder:
    def __init__(self,str_max_dim,num, num_dim, d_per_level, level):
        self.num = num
        self.str_max_dim = str_max_dim
        self.num_dim = num_dim
        self.string_encoder = string_encoder_decoder(str_max_dim)
        self.number_encoder = number_encoder_decoder(num, num_dim, d_per_level, level)




def binary_to_LC(a):
    """
    a: shape (R, C) of 0/1 ints; C must be divisible by 3.
    Returns: shape (R, C//3), where each entry is
             4*bit0 + 2*bit1 + bit2 for each group of 3 bits in the row.
             (i.e. packs 3 binary columns -> one value in [0..7])
    """
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("Input must be 2D")
    if a.shape[1] % 3:
        raise ValueError("Number of columns must be divisible by 3")
    
    triplets = a.reshape(a.shape[0], -1, 3)          # (R, C/3, 3)
    return (triplets[..., 0] << 2) | (triplets[..., 1] << 1) | triplets[..., 2]






def LC_to_binary(a, out_dtype=np.uint8):
    """
    a: shape (R, M) with integer values in [0..7]
       (each value encodes 3 bits: 4*bit0 + 2*bit1 + bit2)
    returns: shape (R, 3*M) array of {0,1} bits,
             where for each original triplet we restore [bit0, bit1, bit2].
    """
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("Input must be 2D")
    if not ((0 <= a).all() and (a <= 7).all()):
        raise ValueError("Values must be in {0,1,2,3,4,5,6,7}")

    bit0 = (a >> 2) & 1  # originally triplets[..., 0]
    bit1 = (a >> 1) & 1  # originally triplets[..., 1]
    bit2 = a & 1         # originally triplets[..., 2]

    out = np.empty((a.shape[0], a.shape[1] * 3), dtype=out_dtype)
    out[:, 0::3] = bit0
    out[:, 1::3] = bit1
    out[:, 2::3] = bit2
    return out






def cross_hamming_general(A: np.ndarray, B: np.ndarray):
    diff = (A[:, None, :] != B[None, :, :])
    D = diff.sum(axis=2)
    return D








def get_table_stats(table,tpc_ds_schema):
    assert table in tpc_ds_schema.keys()
    file_path = '../original_tables/tables/'+table+'.dat'
    df = pd.read_csv(file_path,sep='|',header=None,encoding='ISO-8859-1',dtype=str,names=tpc_ds_schema[table].keys(),index_col=False)
    assert len(tpc_ds_schema[table].keys()) == df.shape[1]    
    table_stats = {}
    for col in tpc_ds_schema[table].keys():
        col_type = tpc_ds_schema[table][col]
        if col_type == "string":
            table_stats[col] = int(df[col].fillna("").str.len().mean())
        if col_type == "numerical": 
            table_stats[col] = (df[col].fillna(0).astype(float).max()+10,df[col].fillna(0).astype(float).min()-10)
    return table_stats




def config_dim(table,tpc_ds_schema,str_base_dim,str_inc_dim,str_max_dim,num_dim,num_str_dim):
    assert table in tpc_ds_schema.keys()
    
    table_stats = get_table_stats(table,tpc_ds_schema)
    
    table_schema = {}
    cur_dim = 0
    for col in tpc_ds_schema[table].keys():
        col_type = tpc_ds_schema[table][col]
        if col_type == "numerical":
            table_schema[col] = (num_dim+num_str_dim,(cur_dim,cur_dim+num_dim+num_str_dim))
            cur_dim += num_dim+num_str_dim
        if col_type == "string":
            dim_allocation = min(str_base_dim + (table_stats[col] * str_inc_dim),str_max_dim)
            dim_allocation = dim_allocation + (dim_allocation % 3)
            table_schema[col] = (dim_allocation,(cur_dim,cur_dim+dim_allocation))
            cur_dim += dim_allocation
    return table_stats, table_schema






def get_config(table,tpc_ds_schema,str_base_dim,str_inc_dim,str_max_dim,num_dim,num_str_dim):
    table_stats = {}
    encoding_plan = {}
    stat, plan = config_dim(table,tpc_ds_schema,str_base_dim,str_inc_dim,str_max_dim,num_dim,num_str_dim)
    table_stats[table] = stat
    encoding_plan[table] = plan
    return table_stats, encoding_plan




def count_lines(path, chunk_size=1024*1024):
    lines = 0
    with open(path, 'rb') as f:
        while chunk := f.read(chunk_size):
            lines += chunk.count(b'\n')
    return lines



def encode_table(encoder,table,tpcds_schema,table_stats,encoding_plan,num_str_dim):
    assert table in tpcds_schema.keys()
    file_path = '../sample_tpcds_table/table/'+table+'.dat'
    num_record = count_lines(file_path)
    total_dim = sum([int(encoding_plan[table][f][0]) for f in tpcds_schema[table].keys()])
    
    table_encoding = np.zeros((num_record, total_dim), dtype='bool')
    
    print("Pre-allocate: ", table, (num_record,total_dim))
    col_name = list(tpcds_schema[table].keys())
    
    with open(file_path, 'r') as file:
        record_counter = 0
        for line in file:
            fields = line.split('|')[:-1]
            record_encoding = []
            for i in range(len(fields)):
                cell = fields[i]
                cell_name = col_name[i]
                cell_type = tpcds_schema[table][cell_name]
                dim_allocation = encoding_plan[table][cell_name][0]
                try:
                    if cell_type == "string":
                        cell = cell if len(cell)<=140 else cell[:140]
                        record_encoding.append((encoder.string_encoder.encode(str(cell),dim_allocation)>0).astype('bool'))
                    if cell_type == "numerical":
                        _max, _min = table_stats[table][cell_name]
                        cell = '' if cell == '' else float(cell)
                        arr1 = (encoder.number_encoder.encode(cell,_min,_max)>0).astype('bool')
                        arr2 = (encoder.string_encoder.encode(str(cell),num_str_dim)>0).astype('bool')
                        record_encoding.append(np.concatenate((arr1,arr2)))
                except Exception as e:
                    print("Error: ", table, cell_name, cell_type, dim_allocation, cell)
                    print(f"Msg: {e}")
            table_encoding[record_counter] = np.hstack(record_encoding)
            record_counter+=1
    return table_encoding




def encode_tables(table_list,encoder,tpcds_schema,table_stats,encoding_plan,num_str_dim):
    for table in table_list:
        table_enc = encode_table(encoder,table,tpcds_schema,table_stats,encoding_plan,num_str_dim)
        assert table_enc.dtype == 'bool'
        table_enc = binary_to_LC(table_enc)
        np.savez_compressed('./encoded_tables/'+table+'.npz', data=table_enc, shape=table_enc.shape)
        print(table," processed", table_enc.shape)
        del table_enc




















