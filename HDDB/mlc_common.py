from common import *
import numpy as numpy




class mlc_common():
	def __init__(self, encoder):
		self.ch = encoder.string_encoder.ch
		self.char_to_index = encoder.string_encoder.char_to_index
		self.alp = binary_to_LC((encoder.string_encoder.alp>0).astype('int'))
		self.I_unrolled = self.alp[len(self.ch)+1:]



		self.mlc_xor_mapping = np.array([
			[0, 1, 2, 3, 4, 5, 6, 7],
		    [1, 0, 3, 2, 5, 4, 7, 6],
		    [2, 3, 0, 1, 6, 7, 4, 5],
		    [3, 2, 1, 0, 7, 6, 5, 4],
		    [4, 5, 6, 7, 0, 1, 2, 3],
		    [5, 4, 7, 6, 1, 0, 3, 2],
		    [6, 7, 4, 5, 2, 3, 0, 1],
		    [7, 6, 5, 4, 3, 2, 1, 0]
		    ], dtype=np.uint8)


		self.range_hvs = binary_to_LC((encoder.number_encoder.range_hvs>0).astype('int'))
		self.nan = binary_to_LC(np.array([(encoder.number_encoder.nan>0).astype('int')]))




	def string_decode(self, mlc_vec):

		working_dim = len(mlc_vec)
		working_alp = self.alp[:,:working_dim]
		working_I_unrolled = self.I_unrolled[:,:working_dim]

		unbinded = self.mlc_xor_mapping[np.repeat(np.expand_dims(mlc_vec, axis=0), 150, axis=0), working_I_unrolled]
		ids = np.argmax(cross_hamming_general(unbinded,working_alp[:13]),axis=1)

		try:
			cut_off = np.where(ids == len(self.ch))[0][0]
			ids = ids[:cut_off]
		except:
			pass
		string = ''.join(self.ch[ids.get()])

		return string



	def sims(self,a,b):
		sim = (a == b).astype('int').sum(axis=1)
		return sim















































