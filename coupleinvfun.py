
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import pygimli.meshtools as mt
import pygimli as pg
import petrorelationship as petroship
import random

def createGradientModel2D(data, mesh, vTop, vBot, logic, x1, y1, x2, y2):

    p = np.polyfit(pg.x(data), pg.y(data), deg=1)  # slope-intercept form
    n = np.asarray([-p[0], 1.0])  # normal vector
    nLen = np.sqrt(np.dot(n, n))

    x = pg.x(mesh.cellCenters())
    x = x[logic]
    z = pg.y(mesh.cellCenters())
    z = z[logic]
    
    zmin = np.interp(x, x1, y1)
    zmax = np.interp(x, x2, y2)
    

    z = (z - zmin)/(zmax - zmin)
    pos = np.column_stack((x, z))

#     d = np.array([np.abs(np.dot(pos[i, :], n) - p[1]) / nLen
#                   for i in range(pos.shape[0])])
              

    return np.interp(z.array(), [0.0, 1.0], [vTop, vBot])
    
    
    
def sepcificline(meshall,out,Xpos,Depth):

    xyzpos = np.zeros((meshall.cellCount(),3))
    for c in meshall.cells():
        xyzpos[c.id(),:] = c.center().array()

    x1,y1= np.mgrid[Xpos[0]:Xpos[0]+1:1j, Depth[0]:0:400j]
    grid_z1 = griddata(xyzpos[:,0:2], out, (x1, y1), method='linear')
    x2,y2= np.mgrid[Xpos[1]:Xpos[1]+1:1j, Depth[1]:0:400j]
    grid_z2 = griddata(xyzpos[:,0:2], out, (x2, y2), method='linear')
    x3,y3= np.mgrid[Xpos[2]:Xpos[2]+1:1j, Depth[2]:0:400j]
    grid_z3 = griddata(xyzpos[:,0:2], out, (x3, y3), method='linear')

    plt.figure(3)
    plt.subplot(1,3,1)
    plt.plot(grid_z1[0],y1[0])
    plt.title('x='+str(Xpos[0]))
    plt.ylabel('Elevation(m)')
    plt.xlabel('x(m)')
    plt.subplot(1,3,2)
    plt.plot(grid_z2[0],y2[0])
    plt.title('x='+str(Xpos[1]))
    plt.ylabel('Elevation(m)')
    plt.xlabel('x(m)')
    plt.subplot(1,3,3)
    plt.plot(grid_z3[0],y3[0])
    plt.title('x='+str(Xpos[2]))
    plt.ylabel('Elevation(m)')
    plt.xlabel('x(m)')
    plt.show()
    
    np.savetxt('velcontour1.txt',np.hstack((grid_z1[0],y1[0])))
    np.savetxt('velcontour2.txt',np.hstack((grid_z2[0],y2[0])))
    np.savetxt('velcontour3.txt',np.hstack((grid_z3[0],y3[0])))
    return xyzpos




def boundary(out,Cvout,xyzpos,value1,value2,firstarea_section,
                        secondarea_section,interval,nrangeinte,flag=0):
    out2 = out[np.array(Cvout,dtype=bool)]
    xyzpos2 = xyzpos[np.array(Cvout,dtype=bool)]
    clusters = np.zeros((out2.shape))

    if flag==0: 
        clusters[out2<=value1] = 0
        clusters[np.logical_and(out2>value1,out2<=value2)] = 1
        clusters[out2>value2] = 2    
    else:
        clusters[out2>value2] = 2
        clusters[np.logical_and(np.logical_and(out2>value1,out2<=value2),xyzpos2[:,1]>-5)] = 1
        clusters[out2<=value1] = 0
        
    cluster1pos = xyzpos2[clusters==0]
    cluster2pos = xyzpos2[clusters==2]    



    nrange = np.arange(firstarea_section[0],firstarea_section[1],interval)
    nrange2 = np.arange(secondarea_section[0],secondarea_section[1],interval)

    loc1 = []
    for i in range(nrange.shape[0]-1):
        set = cluster1pos[np.logical_and(cluster1pos[:,0]>nrange[i],cluster1pos[:,0]<=nrange[i+1]),0:2]
        if set.size != 0:
            yset = set[:,1]
            min_index = np.argmin(yset)
            loc1.append(set[min_index])
    loc1 = np.array(loc1)

    loc2 = []
    for i in range(nrange2.shape[0]-1):
        set = cluster2pos[np.logical_and(cluster2pos[:,0]>nrange2[i],cluster2pos[:,0]<=nrange2[i+1]),0:2]
        if set.size != 0:

            yset = set[:,1]
            max_index = np.argmax(yset)
            loc2.append(set[max_index])
    loc2 = np.array(loc2)

    loc1new = np.interp(nrangeinte, loc1[:,0], loc1[:,1])
    loc2new = np.interp(nrangeinte, loc2[:,0], loc2[:,1])
    plt.figure(4)
    plt.scatter(xyzpos2[clusters == 0,0],xyzpos2[clusters == 0,1])
    plt.scatter(xyzpos2[clusters == 1,0],xyzpos2[clusters == 1,1])
    plt.scatter(xyzpos2[clusters == 2,0],xyzpos2[clusters == 2,1])
    plt.plot(nrangeinte,loc1new,color='k')
    plt.plot(nrangeinte,loc2new,color='k')
    plt.ylabel('Elevation(m)')
    plt.xlabel('x(m)')
    plt.show()

    return loc1new,loc2new


def constraintinvpara(ertData,nrangeinte,loc1new,loc2new,paraBoundary=2,boundary=1):
    #paraBoundary = 5
    geo = pg.meshtools.createParaMeshPLC(ertData, quality=31, paraMaxCellSize=5,
                                         paraBoundary=paraBoundary,paraDepth = 22.0,
                                         boundary=boundary,boundaryMaxCellSize=200)

    locall1 = np.vstack((nrangeinte,loc1new))
    locall2 = np.vstack((nrangeinte,loc2new))

    locall1 = locall1.T
    locall2 = locall2.T


    input1 = np.vstack((locall1,
                        np.array([[locall1[-1][0]+paraBoundary,locall1[-1][1]]])))
                                  
    input1 = np.vstack((np.array([locall1[0][0]-paraBoundary,locall1[0][1]]),input1))
     
    Depth = -(ertData.sensors().array()[0][0]+ertData.sensors().array()[-1][0])*0.4 + np.min(ertData.sensors().array()[0][1])  

    input2 = np.vstack((locall2,
                        np.array([[locall2[-1][0]+paraBoundary,locall2[-1][1]]])))

    input2 = np.vstack((np.array([locall2[0][0]-paraBoundary,locall2[0][1]]),input2))


    #input2 = np.vstack((input1[::-1],input2))
    #input2 = np.vstack((np.array([locall2[0][0]-paraBoundary,Depth]),input2))

    #input2 = np.vstack((input2,np.array([locall2[-1][0],Depth+5])))

    line1 = mt.createPolygon(input1.tolist(), isClosed=False,
                              interpolate='linear')

    line2 = mt.createPolygon(input2.tolist(), isClosed=False,
                              interpolate='linear', marker=4)

    geo1 = geo + line2 +line1


    # ax, _ = pg.show(geo1)
    # ax.set_xlim(-40, 40)
    # ax.set_ylim(-40, 0)


    meshafter = pg.meshtools.createMesh(geo1, quality=31)
    
    markerall = []
    ccc = meshafter.cellMarkers()
    ccc = pg.utils.sparseMatrix2coo(ccc)
    markerall2 = ccc.todense()
    markerall2 = np.array(markerall2)
    
    #print(markerall2)
    
    for i in range(meshafter.cellCount()):
        nodepos = meshafter.cell(i).center().x()
        if nodepos>ertData.sensors()[0][0]-paraBoundary and nodepos<ertData.sensors()[-1][0]+paraBoundary:
            locstd = np.interp(nodepos, input1[:,0], input1[:,1])
            locstd2 = np.interp(nodepos, input2[:,0], input2[:,1])
            if abs(meshafter.cell(i).center().y())<abs(locstd):
                markerall.append(3)
            elif abs(meshafter.cell(i).center().y())<abs(locstd2):
                markerall.append(0)
            else:
                markerall.append(markerall2[0,i])
        else:
            markerall.append(1)

    markerall = np.array(markerall)

    marker_ert = markerall.copy()
    marker_ert[markerall==3] = 2
    marker_ert[markerall==0] = 2
    meshafter.setCellMarkers(marker_ert)

    # ax, _ = pg.show(meshafter,markers=True)
    # ax.set_xlim(-40, 40)
    # ax.set_ylim(-30, 0)

    return markerall, meshafter



def petroMC(iter1,modnum,petrorange,markerall,Mall):


    markerall2 = np.zeros((1,modnum))
    markerall2 = markerall[markerall!=1].copy()

    modelsall = np.zeros((iter1,modnum))
    modelsall2 = np.zeros((iter1,modnum))
    for i in range(iter1):
        porosity2 = petroship.proassign(markerall2,
                                        pro1=random.uniform(petrorange.porosity[0], petrorange.porosity[1]),
                                        pro2=random.uniform(petrorange.porosity[2], petrorange.porosity[3]),
                                        pro3=random.uniform(petrorange.porosity[4], petrorange.porosity[5]))
        #print(porosity2)
        n_model2 = petroship.proassign(markerall2,
                                       pro1=random.uniform(petrorange.n_model[0], petrorange.n_model[1]),
                                       pro2=random.uniform(petrorange.n_model[2], petrorange.n_model[3]),
                                       pro3=random.uniform(petrorange.n_model[4], petrorange.n_model[5]))

        sigmas2 = petroship.proassign(markerall2,
                                           pro1=random.uniform(petrorange.sigmas[0], petrorange.sigmas[1]),
                                           pro2=random.uniform(petrorange.sigmas[2], petrorange.sigmas[3]),
                                           pro3=random.uniform(petrorange.sigmas[4], petrorange.sigmas[5]))
                                           
        rhos2 = petroship.proassign(markerall2,
                                           pro1=random.uniform(petrorange.rhos[0], petrorange.rhos[1]),
                                           pro2=random.uniform(petrorange.rhos[2], petrorange.rhos[3]),
                                           pro3=random.uniform(petrorange.rhos[4], petrorange.rhos[5]))        
                                           
        mod = Mall
        
        #### make true the sigma > 0
        tempsigmas2 = sigmas2[markerall2==3].copy()
        temprhos2 = rhos2[markerall2==3].copy()
        
        delta_value = 1/temprhos2- tempsigmas2
        minsigmas2 = np.min([petrorange.sigmas[0],petrorange.sigmas[1]])
        
        tempsigmas2[delta_value<=0] = np.random.uniform(minsigmas2,1/temprhos2[delta_value<=0])
        
        sigmas2[markerall2==3] = tempsigmas2
        
        ##### only change regolith 
        rhos2[markerall2==3] = 1/(1/rhos2[markerall2==3] - sigmas2[markerall2==3])
        
        #print(sigmas2.shape)
        S_2_rho = petroship.InvArchieSrhos2(rho=mod,rhos=rhos2, n=n_model2, a=petrorange.a_model,sigma_sur=sigmas2)
        #print(S_2_rho.shape)
        #S_2_rho = petroship.InvArchieSrhos(rho=mod,rhos = rhos2, n = n_model2, a=petrorange.a_model,sigmas=sigmas2)
        modelsall[i] = S_2_rho*porosity2
        modelsall2[i] = porosity2

    return modelsall, modelsall2

def petroMCtt(iter1,modnum,petrorange,Mall):


    modelsall = np.zeros((iter1,modnum))
    modelsall2 = np.zeros((iter1,modnum))
    for i in range(iter1):
        porosity2 = np.ones(modnum)*random.uniform(petrorange.porosity[0], petrorange.porosity[1])

        #print(porosity2)
        n_model2 = np.ones(modnum)*random.uniform(petrorange.n_model[0], petrorange.n_model[1])

        sigmas2 = np.ones(modnum)*random.uniform(petrorange.sigmas[0], petrorange.sigmas[1])
        
        rhos2 = np.ones(modnum)*random.uniform(petrorange.rhos[0], petrorange.rhos[1])      
                                           
        mod = Mall
        
        #### make true the sigma > 0
        tempsigmas2 = sigmas2.copy()
        temprhos2 = rhos2.copy()
        
        delta_value = 1/temprhos2- tempsigmas2
        minsigmas2 = np.min([petrorange.sigmas[0],petrorange.sigmas[1]])
        
        tempsigmas2[delta_value<=0] = np.random.uniform(minsigmas2,1/temprhos2[delta_value<=0])
        
        sigmas2 = tempsigmas2.copy()
        
        ##### change whole area, we don't know where is the regolith boundary in traditional
        rhos2 = 1/(1/rhos2- sigmas2)
                
        
        #print(sigmas2.shape)

        S_2_rho = petroship.InvArchieSrhos2(rho=mod,rhos=rhos2, n=n_model2, a=petrorange.a_model,sigma_sur=sigmas2)
        S_2_rho = np.array(S_2_rho)
        S_2_rho = S_2_rho.reshape((-1,))
        #print(S_2_rho.shape)
        
        modelsall[i] = S_2_rho*porosity2
        modelsall2[i]  = porosity2
    return modelsall,modelsall2
    

def petroMC2(iter1,mesh2,mod,petrorange,ttData,meshmarker):
    modelsall = np.zeros((iter1,mod.shape[0]))
    for i in range(iter1):
    
        x1 = np.load('x1.npy')
        x2 = np.load('x2.npy')
        y1 = np.load('y1.npy')
        y2 = np.load('y2.npy')
        
        senpos = np.array (ttData.sensorPositions())
        porosity1 = createGradientModel2D(ttData, mesh2, petrorange.porosity[0], 
                                            petrorange.porosity[1], meshmarker==0,senpos[:,0],senpos[:,1],x1,y1)
       
        porosity2 = createGradientModel2D(ttData, mesh2, petrorange.porosity[2],petrorange.porosity[3],             meshmarker==2,x1,y1,x2,y2)

        porosity3 = pg.solver.parseArgToArray([[0, 0.05], [2, 0.05], [1,random.uniform(petrorange.porosity[4], petrorange.porosity[5])]],
                                              mesh2.cellCount(), mesh2)
        porosity = porosity3.array()
        porosity[meshmarker==0]=porosity1
        porosity[meshmarker==2]=porosity2
        porosity[meshmarker==1]=porosity3[meshmarker==1]                                   
                                                
        n_m = pg.solver.parseArgToArray([[0, random.uniform(0, 1)], 
                                         [2, random.uniform(0, 1)], 
                                         [1, random.uniform(0, 1)]],
                                         mesh2.cellCount(), mesh2)
                                         
        m_model = pg.solver.parseArgToArray([[0, random.uniform(petrorange.m_model[0], petrorange.m_model[1])], 
                                             [2, random.uniform(petrorange.m_model[2], petrorange.m_model[3])], 
                                             [1, random.uniform(petrorange.m_model[4], petrorange.m_model[5])]],
                                              mesh2.cellCount(), mesh2)                                

        S_2_rho = petroship.InvArchieS(rho=mod, rFluid=50, phi=porosity,
                                       m=m_model, n = m_model + n_m, a=1,sigmas=0)
                                       
        modelsall[i] = S_2_rho*porosity

    return modelsall


def petroMC3(iter1,mesh2,mod,petrorange):
    modelsall = np.zeros((iter1,mod.shape[0]))
    for i in range(iter1):
    
        porosity = pg.solver.parseArgToArray([[0, random.uniform(petrorange.porosity[0], petrorange.porosity[1])], 
                                                [2, random.uniform(petrorange.porosity[0], petrorange.porosity[1])], 
                                                [1, random.uniform(petrorange.porosity[0], petrorange.porosity[1])]],
                                                mesh2.cellCount(), mesh2)
                                                
        n_m = pg.solver.parseArgToArray([[0, random.uniform(0, 1)], 
                                         [2, random.uniform(0, 1)], 
                                         [1, random.uniform(0, 1)]],
                                         mesh2.cellCount(), mesh2)
                                         
        m_model = pg.solver.parseArgToArray([[0, random.uniform(petrorange.m_model[0], petrorange.m_model[1])], 
                                             [2, random.uniform(petrorange.m_model[0], petrorange.m_model[1])], 
                                             [1, random.uniform(petrorange.m_model[0], petrorange.m_model[1])]],
                                              mesh2.cellCount(), mesh2)                                

        S_2_rho = petroship.InvArchieS(rho=mod, rFluid=50, phi=porosity,
                                       m=m_model, n = m_model + n_m, a=1,sigmas=0)
                                       
        modelsall[i] = S_2_rho*porosity

    return modelsall

def petroMC4(iter1,mesh2,mod,petrorange):
    modelsall = np.zeros((iter1,mod.shape[0]))
    for i in range(iter1):
    
        porosity = pg.solver.parseArgToArray([[0, random.uniform(petrorange.porosity[0], petrorange.porosity[1])], 
                                                [2, random.uniform(petrorange.porosity[2], petrorange.porosity[3])], 
                                                [1, random.uniform(petrorange.porosity[4], petrorange.porosity[5])]],
                                                mesh2.cellCount(), mesh2)
                                                
        n_m = pg.solver.parseArgToArray([[0, random.uniform(0, 1)], 
                                         [2, random.uniform(0, 1)], 
                                         [1, random.uniform(0, 1)]],
                                         mesh2.cellCount(), mesh2)
                                         
        m_model = pg.solver.parseArgToArray([[0, random.uniform(petrorange.m_model[0], petrorange.m_model[1])], 
                                             [2, random.uniform(petrorange.m_model[2], petrorange.m_model[3])], 
                                             [1, random.uniform(petrorange.m_model[4], petrorange.m_model[5])]],
                                              mesh2.cellCount(), mesh2)                                

        S_2_rho = petroship.InvArchieS(rho=mod, rFluid=50, phi=porosity,
                                       m=m_model, n = m_model + n_m, a=1,sigmas=0)
                                       
        modelsall[i] = S_2_rho*porosity

    return modelsall
    
    
def petroinvMC(ertData,ert,petrorange1,markerall2,meshafter):
    #ss = []
    #for i in range(iter1):
    ERT = ert.ERTManager(verbose=0)
    petro1 = petroship.petroclass

    porosity2 = petroship.proassign(markerall2,
                                        pro1=random.uniform(petrorange1.porosity[0], petrorange1.porosity[1]),
                                        pro2=random.uniform(petrorange1.porosity[2], petrorange1.porosity[3]),
                                        pro3=random.uniform(petrorange1.porosity[4], petrorange1.porosity[5]))
    #print(porosity2)
    m_model2 = petroship.proassign(markerall2,
                                       pro1=random.uniform(petrorange1.m_model[0], petrorange1.m_model[1]),
                                       pro2=random.uniform(petrorange1.m_model[2], petrorange1.m_model[3]),
                                       pro3=random.uniform(petrorange1.m_model[4], petrorange1.m_model[5]))
                                       
    sigmas2 = petroship.proassign(markerall2,
                                       pro1=random.uniform(petrorange1.sigmas[0], petrorange1.sigmas[1]),
                                       pro2=random.uniform(petrorange1.sigmas[2], petrorange1.sigmas[3]),
                                       pro3=random.uniform(petrorange1.sigmas[4], petrorange1.sigmas[5]))                                   
    petro1.porosity = porosity2
    petro1.m_model = m_model2
    petro1.sigmas = sigmas2
    petro1.rFluid = petrorange1.rFluid
    petro1.a_model = petrorange1.a_model
    petro1.n_model = m_model2+0.5

    ERTPetro = petroship.PetroInversionManager2(petro=petro1, mgr=ERT)
    satERT = ERTPetro.invert(ertData, mesh=meshafter, limits=[0.01, 0.5], lam=200, verbose=False, maxIter=20)
    ERTPetro.inv.reset()
    ERT.inv.reset()

    ww = satERT.array().copy()
    
    watercontent = ww

    del satERT
    del ERTPetro
    del ERT
    del petro1
    del porosity2
    del m_model2

    return watercontent
