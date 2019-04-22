#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 12:35:19 2019

@author: saurabh
"""

import pandas as pd
import numpy as np
import datetime
import time
from sklearn.linear_model import LinearRegression
import csv
import time


df = pd.read_csv('/Users/saurabh/Desktop/train_values.csv')
print(df.shape)
df[:5000].to_csv('/Users/saurabh/Downloads/DrivenData/train_sample.csv')

df = df.loc[df['phase'] != 'final_rinse']
print(df.shape)

def func(row):
    if row['phase'] == 'pre_rinse':
        return 1
    elif row['phase'] =='caustic':
        return 2
    elif row['phase'] =='intermediate_rinse':
        return 3
    elif row['phase'] =='acid':
        return 4
    else:
        return 5

df['phase_num'] = df.apply(func, axis=1)

# randomly select 80% of phases to keep
df['process_phase'] = df.process_id.astype(str) + '_' + df.phase.astype(str)
process_phases = df.process_phase.unique()

rng = np.random.RandomState(2402)
to_keep = rng.choice(
                process_phases,
                size=np.int(len(process_phases) * 0.8),
                replace=False)

train_limited = df[df.process_phase.isin(to_keep)]
print(train_limited.shape)
train_limited.to_csv('/Users/saurabh/Downloads/DrivenData/train_selected.csv')

# subset labels to match our training data
#train_labels = train_labels.loc[train_limited.process_id.unique()]

train_limited[:10000].to_csv('/Users/saurabh/Desktop/sample_train_2.csv', sep = ',')

df2 = pd.read_csv('/Users/saurabh/Downloads/DrivenData/train_selected.csv')
#mean1 = df2.groupby(['process_id']).agg({"phase_num": ["min", "max", np.mean, np.var],"tank_temperature_acid":["count"], "timestamp":["min", "max"] })
#mean1.columns = ["_".join(x) for x in mean1.columns.ravel()]
#print(mean1[:2])

def diffdates(mean1):
    #Date format: %Y-%m-%d %H:%M:%S
    return (time.mktime(time.strptime(mean1["timestamp_max"],"%Y-%m-%d %H:%M:%S")) -
               time.mktime(time.strptime(mean1["timestamp_min"], "%Y-%m-%d %H:%M:%S")))/60
    
    
#Pipleines

#process_id = df2.groupby(["process_id","pipeline"]).count()
#print(process_id)
df2[['supply_pump', 'supply_pre_rinse', 'supply_caustic', 'return_caustic', 'supply_acid', 'return_acid', 'supply_clean_water', 'return_recovery_water', 'return_drain', 'object_low_level', 'tank_lsh_caustic', 'tank_lsh_clean_water' ]] = df2[['supply_pump', 'supply_pre_rinse', 'supply_caustic', 'return_caustic', 'supply_acid', 'return_acid', 'supply_clean_water', 'return_recovery_water', 'return_drain', 'object_low_level', 'tank_lsh_caustic', 'tank_lsh_clean_water' ]] .astype(int)

mean1 = df2.groupby(['process_id']).agg( {"phase_num": ["min",  "max", "mean", "count"],
"timestamp":["min", "max"],
"supply_flow": ["min",  "max", "mean", "var" ],

"supply_pressure": ["min",  "max", "mean", "var" ],
"return_temperature": ["min",  "max", "mean", "var" ],
"return_conductivity": ["min",  "max", "mean", "var" ],
"return_turbidity": ["min",  "max", "mean", "var" ],
"return_flow": ["min",  "max", "mean", "var" ],
"supply_pump": ["min",  "max", "mean", "var" ],
"supply_pre_rinse": ["min",  "max", "mean", "var" ],
"supply_caustic": ["min",  "max", "mean", "var" ],
"return_caustic": ["min",  "max", "mean", "var" ],
"supply_acid": ["min",  "max", "mean", "var" ],
"return_acid": ["min",  "max", "mean", "var" ],
"supply_clean_water": ["min",  "max", "mean", "var" ],
"return_recovery_water": ["min",  "max", "mean", "var" ],
"return_drain": ["min",  "max", "mean", "var" ],
"object_low_level": ["min",  "max", "mean", "var" ],
"tank_level_pre_rinse": ["min",  "max", "mean", "var" ],
"tank_level_caustic": ["min",  "max", "mean", "var" ],
"tank_level_acid": ["min",  "max", "mean", "var" ],
"tank_level_clean_water": ["min",  "max", "mean", "var" ],
"tank_temperature_pre_rinse": ["min",  "max", "mean", "var" ],
"tank_temperature_caustic": ["min",  "max", "mean", "var" ],
"tank_temperature_acid": ["min",  "max", "mean", "var" ],
"tank_concentration_caustic": ["min",  "max", "mean", "var" ],
"tank_concentration_acid": ["min",  "max", "mean", "var" ],
"tank_lsh_caustic": ["min",  "max", "mean", "var" ],
"tank_lsh_acid": ["min",  "max", "mean", "var" ],
"tank_lsh_clean_water": ["min",  "max", "mean", "var" ],
"tank_lsh_pre_rinse": ["min",  "max", "mean", "var" ]
 }).reset_index()

mean1.columns = ["_".join(x) for x in mean1.columns.ravel()]
mean1['supply_flow_cv']= mean1['supply_flow_var']/mean1['supply_flow_mean']
mean1['supply_pressure_cv']= mean1['supply_pressure_var']/mean1['supply_pressure_mean']
mean1['return_temperature_cv']= mean1['return_temperature_var']/mean1['return_temperature_mean']
mean1['return_turbidity_cv']= mean1['return_turbidity_var']/mean1['return_turbidity_mean']
mean1['return_flow']= mean1['return_flow_var']/mean1['return_flow_mean']
mean1['tank_level_pre_rinse_cv']= mean1['tank_level_pre_rinse_var']/mean1['tank_level_pre_rinse_mean']
mean1['tank_level_caustic_cv']= mean1['tank_level_caustic_var']/mean1['tank_level_caustic_mean']
mean1['tank_level_acid_cv']= mean1['tank_level_acid_var']/mean1['tank_level_acid_mean'] 
mean1['tank_level_clean_water_cv']= mean1['tank_level_clean_water_var']/mean1['tank_level_clean_water_mean'] 
mean1['tank_temperature_pre_rinse_cv']= mean1['tank_temperature_pre_rinse_var']/mean1['tank_temperature_pre_rinse_mean']
mean1['tank_temperature_caustic_cv']= mean1['tank_temperature_caustic_var']/mean1['tank_temperature_caustic_mean'] 
mean1['tank_temperature_acid_cv']= mean1['tank_temperature_acid_var']/mean1['tank_temperature_acid_mean'] 
mean1['tank_concentration_caustic_cv']= mean1['tank_concentration_caustic_var']/mean1['tank_concentration_caustic_mean'] 
mean1['tank_concentration_acid_cv']= mean1['tank_concentration_acid_var']/mean1['tank_concentration_acid_mean'] 

mean1['tank_lsh_caustic_cv']= mean1['tank_lsh_caustic_var']/(1+mean1['tank_lsh_caustic_mean'])
mean1['tank_lsh_acid_cv']= mean1['tank_lsh_acid_var']/(1+mean1['tank_lsh_acid_mean'])
mean1['tank_lsh_clean_water_cv']= mean1['tank_lsh_clean_water_var']/(1+mean1['tank_lsh_clean_water_mean'])
mean1['tank_lsh_pre_rinse_cv']= mean1['tank_lsh_pre_rinse_var']/(1+mean1['tank_lsh_pre_rinse_mean'] )
#mean1['_cv']= mean1['_var']/mean1[''] 


mean1["time_diff"] = mean1.apply(diffdates, axis = 1)
mean1 = mean1.drop(["timestamp_min", "timestamp_max"], axis = 1)

print(type(mean1))
print(mean1.shape)
print(mean1[:5])
mean1.to_csv('/Users/saurabh/Downloads/DrivenData/mean1.csv', sep = ',')

pos = 0
colname = mean1.columns[pos]
print(colname)
'''
with open('/Users/saurabh/Desktop/mean1.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader, None)
    your_list = list(reader)
'''
#print(len(your_list), len(your_list[0]))

#Labels
df_label = pd.read_csv('/Users/saurabh/Desktop/train_labels.csv')
print(type(df_label))
#df_label = df_label.groupby(['process_id']).sum()
print(df_label.shape)
#print(df_label[:5])
#print(mean1[:5])

df_merge = pd.merge(mean1, df_label, left_on = 'process_id_', right_on = 'process_id', how = 'inner')
print(df_merge[:5])
print(df_merge.shape)
df_merge = df_merge.drop(["process_id"], axis = 1)
print(df_merge[:5])
print(df_merge.shape)

df_merge.to_csv('/Users/saurabh/Downloads/DrivenData/devdata_1.csv', sep = ',')