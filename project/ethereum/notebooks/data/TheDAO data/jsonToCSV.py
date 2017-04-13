#!/bin/usr/env python2.7

import json
import io
import numpy as np
import pandas as pd
from ctypes import *

def hexToFloat(hex):
	flt = cast(pointer(c_int(int(hex,16))), POINTER(c_float))
	return flt.contents.value

def weiToEther(wei):
	ether = wei/(10**18)
	return ether

with open('daoDrainList.json') as data_file:
	data_loaded = json.load(data_file)
#print data_loaded

index = ['address','balance','extraBalance','extraBalanceAccount']

TheDaoDF = pd.DataFrame(data=np.random.randn(len(data_loaded), len(index)), columns=index)
index = 0

# print convert("41973333")
# print convert("0x4c6679d9d9b95a4e08")
# print convert("4c6679d9d9b95a4e08")


# print struct.unpack('!f', '41973333'.decode('hex'))[0]
# print struct.unpack('!f', '0x4c6679d9d9b95a4e08'.decode('hex'))[0]
# print struct.unpack('!f', '1'.decode('hex'))[0]
# print struct.unpack('!f', 'd3ff7771412bbcc9'.decode('hex'))[0]
# print struct.unpack('!f', '3635ce47fabaaa336e'.decode('hex'))[0]
# print struct.unpack('!f', '0'.decode('hex'))[0]
# exit(0)

for DAOAccount in data_loaded:
	TheDaoDF.loc[index, 'address']= DAOAccount['address']
	#TheDaoDF['balance'][index] = '0x' + DAOAccount['balance']
	TheDaoDF.loc[index,'balance'] = weiToEther(hexToFloat(DAOAccount['balance']))
	TheDaoDF.loc[index, 'extraBalance'] = weiToEther(hexToFloat(DAOAccount['extraBalance']))
	#TheDaoDF['extraBalance'][index] = DAOAccount['extraBalance']
	TheDaoDF.loc[index, 'extraBalance'] = DAOAccount['extraBalanceAccount']
	index += 1

TheDaoDF.to_csv('daoDrainList.csv',sep=',',index=False)



exit(0)