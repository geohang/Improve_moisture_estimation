

import multiprocessing
import coupleinvfun as CF
from pygimli.physics import ert
import numpy as np
import pygimli as pg
import petrorelationship as petroship
import os




para = pg.load('para2.bms')
meshafter = pg.load('meshafter2.bms')
markerall = np.load('markerall2.npy')

petrorange = petroship.petrorange
petrorange.porosity = [0.45, 0.3, 0.1, 0.35, 0.02, 0.1]
petrorange.m_model = [1.4, 2.2, 1.0, 1.5, 1.6, 2.5]
petrorange.rFluid = 50
petrorange.n_model = 2
petrorange.a_model = 1.0
petrorange.sigmas = [0, 0, 0, 0, 0, 0]

ertData = ert.load('ertData2.dat')
                                
markerall2 = np.zeros((1,para.cellCount()))
markerall2 = markerall[markerall != 1].copy()




def f(x):
    pid = os.getpid()
    f1 = open(file='./pid/count_pid'+str(x)+'.txt', mode='w')
    f1.write(pid.__str__())
    f1.close()
    return CF.petroinvMC(ertData,ert,petrorange,markerall2,meshafter)

def paraf(ele):
    pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()-4))
    res = pool.map(f,range(ele))
    return res


def main(num=100,name=1):


    
    print('---------------------it is a test in main--------------------')

    cc = paraf(num)
    
    np.save('./Res/Res'+str(name),np.array(cc))
    #import os
    #os._exit(00)




