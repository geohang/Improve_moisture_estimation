

import pygimli as pg
from pygimli.physics import ert  # the module
import numpy as np
import invesfun as inv
from scipy import sparse
from scipy.sparse.linalg import lsqr
import pygimli.meshtools as mt
import matplotlib.colors as mcolors
import pygimli.physics.traveltime as tt
from pygimli.physics import TravelTimeManager
import invclass


#----------------------------load data module----------------------------#
#load ERT data
dataert = ert.load("ERTpo")
datasrt = tt.load("SRTpo")





#----------------------------obtain apparent data module--------------------#
# obtain the apparent resistivity
#dataert['k'] = ert.createGeometricFactors(dataert, numerical=True)
#rhos = data['k']*data['u']/data['i']
rhos = dataert['rhoa']
rhos1 = np.log(rhos.array())
rhos1 = rhos1.reshape(rhos1.shape[0],1)

dobs_s = datasrt['t'].array()

dobs_s = dobs_s.reshape(dobs_s.shape[0],1)

ert1 = ert.ERTManager(dataert)
mgr = TravelTimeManager()



#----------------------------mesh module--------------------#
xpos = []
ypos = []
world2 = mt.createWorld(start=[-21, 0], end=[22, -18], marker=2, worldMarker=True)
for pos in dataert.sensorPositions():
    world2.createNode(pos)
    world2.createNode(pos+[0, -0.1])
    xpos.append(pos.array()[0])
    ypos.append(pos.array()[1])


mesh = mt.createMesh(world2,  quality=31, marker=2, area=3, smooth=[2, 2],paraDX=0.5)
grid = pg.meshtools.appendTriangleBoundary(mesh, marker=1,
                                           xbound=100, ybound=100)
# set grid for ERT
ert1.setMesh(grid)
# set mesh for SRT
mgr.setMesh(mesh)

L_r = 15
mr = invclass.ertinv(rhos1, mesh, grid, L_r, dataert, ert)
mpo2 = invclass.INVArchiephi(np.exp(mr))





L_r = 15
mv,Cv = invclass.srtinv(dobs_s, mesh, L_r, datasrt, mgr, flag=1,vT=500, vB=4000)
mpo4 = invclass.INVWylliephi(np.exp(mv))

L_r = 30
mpo3,C3 = invclass.srtporosity(dobs_s, mesh, L_r, datasrt, mgr, flag=1)




L_r = 30
mpo1 = invclass.ertporosity(rhos1, mesh, grid, L_r, dataert, ert)

dobs_s = np.log(dobs_s)

dataall = np.vstack((rhos1,dobs_s))
L_r = 50
mpo5 = invclass.jointporosity(dataall, mesh, grid, L_r, dataert,datasrt, ert, mgr,flag=2)




ax1, _ = pg.show(ert1.fop.paraDomain, (mpo1[:,0]), cMap='jet', label=pg.unit('res'), hold=True)
ax1, _ = pg.show(ert1.fop.paraDomain, (mpo2[:,0]), cMap='jet', label=pg.unit('res'), hold=True)

ax1, _ = pg.show(ert1.fop.paraDomain, (mpo3[:,0]), coverage=C3, cMap='jet', label=pg.unit('res'))
ax1, _ = pg.show(ert1.fop.paraDomain, (mpo4[:,0]), coverage=Cv, cMap='jet', label=pg.unit('res'))

ax1, _ = pg.show(ert1.fop.paraDomain, (mpo5[:,0]), cMap='jet', label=pg.unit('res'))

ax1, _ = pg.show(ert1.fop.paraDomain, 1./np.exp(mv[:,0]), coverage=Cv, cMap='jet', label=pg.unit('res'), cMin=500, cMax=4000)

s=1

