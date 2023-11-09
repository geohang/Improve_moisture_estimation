

import multiprocessing
import coupleinvfun as CF
from pygimli.physics import ert
import numpy as np
import pygimli as pg
import petrorelationship as petroship
import os




para = pg.load('M_para1.bms')
meshafter = pg.load('M_meshafter1.bms')
markerall = np.load('M_markerall1.npy')

petrorange = petroship.petrorange
petrorange.porosity = [0.45, 0.3, 0.1, 0.35, 0.02, 0.1]
petrorange.m_model = [1.4, 2.2, 1.0, 1.5, 1.6, 2.5]
petrorange.sigmas = [1e-4, 5e-3, 1e-4, 2e-4, 0, 0]
petrorange.rFluid = 50
petrorange.n_model = 2
petrorange.a_model = 1.0

ertData = ert.load('test1.dat')
ertData['err'] = ert.ERTManager().estimateError(ertData, 
                                absoluteUError=0.00005, # 50µV
                                relativeError=0.06)
                                
markerall2 = np.zeros((1,para.cellCount()))
markerall2 = markerall[markerall != 1].copy()




def f(x):
    pid = os.getpid()
    f1 = open(file='./pid/count_pid'+str(x)+'.txt', mode='w')
    f1.write(pid.__str__())
    f1.close()
    return CF.petroinvMC(ertData,ert,petrorange,markerall2,meshafter)

def paraf(ele):
    pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()))
    res = pool.map(f,range(ele))
    return res


def main(num=100,name=1):


    
    print('---------------------it is a test in main--------------------')

    cc = paraf(num)
    
    np.save('Res'+str(name),np.array(cc))
    #import os
    #os._exit(00)




