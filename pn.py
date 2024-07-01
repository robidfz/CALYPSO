import matplotlib.pyplot as plt
import pandas as pd
import sys
import pm4py
from configparser import ConfigParser
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.visualization.petri_net import visualizer as pn_visualizer






data_excel=pd.read_csv('dataset.csv',sep=',')
data_excel=data_excel[data_excel['Detail']!='VALUE']
data_excel.rename(columns={'observation': 'concept:name'}, inplace=True)

event_log = log_converter.apply(data_excel)
# alpha miner
# net, initial_marking, final_marking = alpha_miner.apply(event_log)
net, initial_marking, final_marking = pm4py.discover_petri_net_alpha(event_log)
gviz = pn_visualizer.apply(net, initial_marking, final_marking)
pn_visualizer.view(gviz)

