
import numpy as np
import pygimli as pg
from scipy import sparse
from scipy.sparse.linalg import lsqr
import invesfun as inv
import pygimli.physics.petro as petro
import pygimli.physics.traveltime as tt




def Archiephi(a=1, n=2, rhof=20, phi=0.3,m=2, S=0.7):

    return a*rhof/(phi**m*S**n)

def INVArchiephi(rho, a=1, n=2, rhof=20,m=2, S=0.7):

    return (a*rhof/(rho*S**n))**(1/m)

def Archiephideri(a=1, n=2, rhof=20, phi=0.3,m=2, S=0.7):

    return -m*a*rhof/(phi**(m+1)*S**n)

def Wyllie(phi, sat=0.7, vm=5000, vw=1400, va=300):

    return 1./vm * (1.-phi) + 1./vw * phi * sat + 1./va * phi * (1. - sat)

def Wylliephideri(phi, sat=0.7, vm=5000, vw=1400, va=300):

    return -1./vm + 1./vw * sat + 1./va * (1. - sat) + np.zeros((phi.shape))

def INVWylliephi(V, sat=0.7, vm=5000, vw=1400, va=300):

    return (V-1/vm)/((1-sat)/va+sat/vw-1/vm)




def ertpoF2(fob,xr,mesh):
    xr1 = xr.copy()
    xr1[mesh.cellMarkers() == 2] = xr
    xr1 = Archiephi(phi=xr1)
    rhomodel = pg.matrix.RVector(xr1)

    dr = fob.response(rhomodel)
    dr = np.log(dr)
    return dr

def ertpoJ2(fob,xr,mesh):
    xr1 = xr.copy()
    xr1[mesh.cellMarkers() == 2] = xr
    Dm = Archiephideri(phi=xr1)
    xr1 = Archiephi(phi=xr1)

    rhomodel = pg.matrix.RVector(xr1)
    dr = fob.response(rhomodel)
    fob.createJacobian(rhomodel)
    J = fob.jacobian()
    J = pg.utils.gmat2numpy(J)
    J = (Dm.T)*J
    dr = dr.array()
    J = J/dr.reshape(dr.shape[0],1)
    dr = np.log(dr)
    return dr, J

def srtpoJ2(fob,sm):

    x1 = Wyllie(sm)
    Dm = Wylliephideri(phi=sm)
    dr = fob.response(x1)
    dr = dr.array()
    fob.createJacobian(x1)
    J = fob.jacobian()
    J = pg.utils.sparseMatrix2coo(J)
    J = J.todense()
    J = np.array(J)
    J = (Dm.T)*J
    return dr, J

def srtpoF2(fob,sm):

    x1 = Wyllie(sm)

    dr = fob.response(x1)
    dr = (dr.array())
    return dr

def srtpoJ3(fob,sm):

    x1 = Wyllie(sm)
    Dm = Wylliephideri(phi=sm)
    dr = fob.response(x1)
    dr = dr.array()
    fob.createJacobian(x1)
    J = fob.jacobian()
    J = pg.utils.sparseMatrix2coo(J)
    J = J.todense()
    J = np.array(J)
    J = (Dm.T)*J
    J = J/dr.reshape(dr.shape[0],1)
    dr = np.log(dr)
    return dr, J

def srtpoF3(fob,sm):

    x1 = Wyllie(sm)

    dr = fob.response(x1)
    dr = np.log(dr.array())
    return dr


def srtpoJ(fob,sm):

    x1 = Wyllie(sm)
    fob.createJacobian(x1)
    J = fob.jacobian()
    return J

def jointpoF3(fobert,fobsrt,xr,mesh):
    xr1 = xr.copy()
    xr1[mesh.cellMarkers() == 2] = xr
    xr1 = Archiephi(phi=xr1)
    rhomodel = pg.matrix.RVector(xr1)
    dr_r = fobert.response(rhomodel)
    dr_r = dr_r.array()
    dr_r = np.log(dr_r)
    dr_r = dr_r.reshape(dr_r.shape[0],1)

    sm = xr
    x1 = Wyllie(sm)
    dr_s = fobsrt.response(x1)
    dr_s = dr_s.array()
    dr_s = dr_s.reshape(dr_s.shape[0],1)
    dr = np.vstack((dr_r,dr_s))
    return dr

def jointpoJ3(fobert,fobsrt,xr,mesh):
    xr1 = xr.copy()
    xr1[mesh.cellMarkers() == 2] = xr
    Dm = Archiephideri(phi=xr1)
    xr1 = Archiephi(phi=xr1)

    rhomodel = pg.matrix.RVector(xr1)
    dr_r = fobert.response(rhomodel)
    fobert.createJacobian(rhomodel)
    Jr = fobert.jacobian()
    Jr = pg.utils.gmat2numpy(Jr)
    Jr = (Dm.T)*Jr
    dr_r = dr_r.array()
    Jr = Jr/dr_r.reshape(dr_r.shape[0],1)
    dr_r = np.log(dr_r)
    dr_r = dr_r.reshape(dr_r.shape[0],1)

    sm = xr
    x1 = Wyllie(sm)
    Dm2 = Wylliephideri(phi=sm)
    dr_s = fobsrt.response(x1)
    dr_s = dr_s.array()
    dr_s = dr_s.reshape(dr_s.shape[0],1)
    fobsrt.createJacobian(x1)
    Js = fobsrt.jacobian()
    Js = pg.utils.sparseMatrix2coo(Js)
    Js = Js.todense()
    Js = np.array(Js)
    Js = (Dm2.T)*Js


    J = np.vstack((Jr,Js))
    dr = np.vstack((dr_r,dr_s))
    return dr, J

def jointpoF2(fobert,fobsrt,xr,mesh):
    xr1 = xr.copy()
    xr1[mesh.cellMarkers() == 2] = xr
    xr1 = Archiephi(phi=xr1)
    rhomodel = pg.matrix.RVector(xr1)
    dr_r = fobert.response(rhomodel)
    dr_r = dr_r.array()
    dr_r = np.log(dr_r)
    dr_r = dr_r.reshape(dr_r.shape[0],1)

    sm = xr
    x1 = Wyllie(sm)
    dr_s = fobsrt.response(x1)
    dr_s = dr_s.array()
    dr_s = dr_s.reshape(dr_s.shape[0],1)
    dr_s = np.log(dr_s)
    dr = np.vstack((dr_r,dr_s))
    return dr

def jointpoJ2(fobert,fobsrt,xr,mesh):
    xr1 = xr.copy()
    xr1[mesh.cellMarkers() == 2] = xr
    Dm = Archiephideri(phi=xr1)
    xr1 = Archiephi(phi=xr1)

    rhomodel = pg.matrix.RVector(xr1)
    dr_r = fobert.response(rhomodel)
    fobert.createJacobian(rhomodel)
    Jr = fobert.jacobian()
    Jr = pg.utils.gmat2numpy(Jr)
    Jr = (Dm.T)*Jr
    dr_r = dr_r.array()
    Jr = Jr/dr_r.reshape(dr_r.shape[0],1)
    dr_r = np.log(dr_r)
    dr_r = dr_r.reshape(dr_r.shape[0],1)

    sm = xr
    x1 = Wyllie(sm)
    Dm2 = Wylliephideri(phi=sm)
    dr_s = fobsrt.response(x1)
    dr_s = dr_s.array()
    dr_s = dr_s.reshape(dr_s.shape[0],1)
    fobsrt.createJacobian(x1)
    Js = fobsrt.jacobian()
    Js = pg.utils.sparseMatrix2coo(Js)
    Js = Js.todense()
    Js = np.array(Js)
    Js = (Dm2.T)*Js
    Js = Js/dr_s.reshape(dr_s.shape[0],1)
    dr_s = np.log(dr_s)

    J = np.vstack((Jr,Js))
    dr = np.vstack((dr_r,dr_s))
    return dr, J



def ertinv(rhos1, mesh, grid, L_r, data, ert):

    rhomap = [[2, np.mean(np.exp(rhos1))]]
    rhomodel = pg.solver.parseArgToArray(rhomap, mesh.cellCount(), mesh)

    xr1 = np.log(rhomodel.array())
    xr = xr1[mesh.cellMarkers() == 2]
    xr = xr.reshape(xr.shape[0],1)
    xr_R = xr
    xr_R = xr_R.reshape(xr_R.shape[0],1)


    fob = ert.ERTModelling()
    fob.setData(data)
    fob.setMesh(grid)



    # data weight matrix
    rmag_linear = rhos1
    dobs = rhos1
    dobs = dobs.reshape(dobs.shape[0],1)

    Wd = np.diag(1.0/ np.abs(rmag_linear[:,0] * 0.01))
    WdTwd = Wd.T.dot(Wd)

    # model weighting matrix
    rm = fob.regionManager()
    Ctmp = pg.matrix.RSparseMapMatrix()
    rm.setConstraintType(1)
    rm.fillConstraints(Ctmp)

    Wm_r = pg.utils.sparseMatrix2coo(Ctmp)
    Wm_r = Wm_r.todense()
    Wm_r = L_r*Wm_r
    chi2 = 1
    for i in range(10):
        dr,J = inv.ertforandjac2(fob, xr, mesh)
        dr = dr.reshape(dr.shape[0],1)


        grad_fd = np.dot(np.dot(J.transpose(),WdTwd), (dr-dobs))
        gg = grad_fd + np.dot(Wm_r.transpose().dot(Wm_r),xr-xr_R)

        N11 =np.vstack((Wd.dot(J), Wm_r))

        dataerror = dr - dobs
        gc1 = np.vstack((Wd.dot(dataerror),(Wm_r.dot(xr-xr_R))))
        print('Iterations'+str(i+1))
        print('max error:'+str(np.max(dataerror/dobs)))

        fd = (np.dot(Wd, dataerror)).T.dot(np.dot(Wd,dataerror))

        fm = (Wm_r*(xr-xr_R)).T.dot(Wm_r*(xr-xr_R))

        fc = fm + fd

        chi2new = fd/len(dr)
        print('chi2:'+str(chi2new))
        print('dPhi:'+str(abs(chi2-chi2new)/chi2))
        if chi2new < 1.5 or abs(chi2-chi2new)/chi2 < 0.01:
            print('chi2='+str(fd/len(dr)))
            print('dPhi:'+str(abs(chi2-chi2new)/chi2))
            break

        chi2 = chi2new

        sN11 = sparse.csr_matrix(N11)
        mu_LS = 1
        iarm = 1
        gc2 = np.array(gc1)
        gc = gc2.reshape((-1,))

        d, istop, itn, normr = lsqr(sN11, -gc, damp=0.0, atol=1e-08, btol=1e-08, conlim=100000000.0, iter_lim=None, show=False, calc_var=False, x0=None)[:4]
        d = d.reshape(d.shape[0],1)

        while 1:
            xt = xr + mu_LS*d
            dr1 = inv.ertforward2(fob, xt, mesh)
            dr1 = dr1.reshape(dr1.shape[0],1)
            dataerror = dr1 - dobs
            fd = (np.dot(Wd, dataerror)).T.dot(np.dot(Wd,dataerror))
            fm = (Wm_r*(xt-xr_R)).T.dot(Wm_r*(xt-xr_R))
            ft = fm + fd

            fgoal = fc - 1e-4*mu_LS*(d.T.dot(gg.reshape(gg.shape[0],1)))

            #print('ft'+str(ft))
            #print('fgoal'+str(fgoal))
            if ft < fgoal:
                break
            else:
                iarm = iarm+1
                mu_LS = mu_LS/2

            if iarm > 20:
                pg.boxprint('Line search FAIL EXIT')
                break


        xr = xt
        dr = dr1
        xr[xr > np.log(5000)] = np.log(5000)
        xr[xr < np.log(10)] = np.log(10)


    return xr


def ertporosity(rhos1, mesh, grid, L_r, data, ert):

    rhomap = [[2, 0.5]]
    rhomodel = pg.solver.parseArgToArray(rhomap, mesh.cellCount(), mesh)

    xr1 = rhomodel.array()
    xr = xr1[mesh.cellMarkers() == 2]
    xr = xr.reshape(xr.shape[0],1)
    xr_R = xr
    xr_R = xr_R.reshape(xr_R.shape[0],1)


    fob = ert.ERTModelling()
    fob.setData(data)
    fob.setMesh(grid)



    # data weight matrix
    rmag_linear = rhos1
    dobs = rhos1
    dobs = dobs.reshape(dobs.shape[0],1)

    Wd = np.diag(1.0/ np.abs(rmag_linear[:,0] * 0.01))
    WdTwd = Wd.T.dot(Wd)

    # model weighting matrix
    rm = fob.regionManager()
    Ctmp = pg.matrix.RSparseMapMatrix()
    rm.setConstraintType(1)
    rm.fillConstraints(Ctmp)

    Wm_r = pg.utils.sparseMatrix2coo(Ctmp)
    Wm_r = Wm_r.todense()
    Wm_r = L_r*Wm_r
    chi2 = 1
    for i in range(10):
        dr,J = ertpoJ2(fob, xr, mesh)
        dr = dr.reshape(dr.shape[0],1)


        grad_fd = np.dot(np.dot(J.transpose(),WdTwd), (dr-dobs))
        gg = grad_fd + np.dot(Wm_r.transpose().dot(Wm_r),xr-xr_R)

        N11 =np.vstack((Wd.dot(J), Wm_r))

        dataerror = dr - dobs
        gc1 = np.vstack((Wd.dot(dataerror),(Wm_r.dot(xr-xr_R))))
        print('Iterations'+str(i+1))
        print('max error:'+str(np.max(dataerror/dobs)))

        fd = (np.dot(Wd, dataerror)).T.dot(np.dot(Wd,dataerror))

        fm = (Wm_r*(xr-xr_R)).T.dot(Wm_r*(xr-xr_R))

        fc = fm + fd

        chi2new = fd/len(dr)
        print('chi2:'+str(chi2new))
        print('dPhi:'+str(abs(chi2-chi2new)/chi2))
        if chi2new < 1.5 or abs(chi2-chi2new)/chi2 < 0.01:
            print('chi2='+str(fd/len(dr)))
            print('dPhi:'+str(abs(chi2-chi2new)/chi2))
            break

        chi2 = chi2new

        sN11 = sparse.csr_matrix(N11)
        mu_LS = 1
        iarm = 1
        gc2 = np.array(gc1)
        gc = gc2.reshape((-1,))

        d, istop, itn, normr = lsqr(sN11, -gc, damp=0.0, atol=1e-08, btol=1e-08, conlim=100000000.0, iter_lim=None, show=False, calc_var=False, x0=None)[:4]
        d = d.reshape(d.shape[0],1)

        while 1:
            xt = xr + mu_LS*d
            dr1 = ertpoF2(fob, xt, mesh)
            dr1 = dr1.reshape(dr1.shape[0],1)
            dataerror = dr1 - dobs
            fd = (np.dot(Wd, dataerror)).T.dot(np.dot(Wd,dataerror))
            fm = (Wm_r*(xt-xr_R)).T.dot(Wm_r*(xt-xr_R))
            ft = fm + fd

            fgoal = fc - 1e-4*mu_LS*(d.T.dot(gg.reshape(gg.shape[0],1)))

            #print('ft'+str(ft))
            #print('fgoal'+str(fgoal))
            if ft < fgoal:
                break
            else:
                iarm = iarm+1
                mu_LS = mu_LS/2

            if iarm > 20:
                pg.boxprint('Line search FAIL EXIT')
                break


        xr = xt
        dr = dr1
        xr[xr > 1] = 1
        xr[xr < 0.01] = 0.01

    return xr




def srtinv(rhos1, mesh, L_r, data, mgr,flag=1, vT=200, vB=1000):


    sm = tt.createGradientModel2D(data,mesh, vTop=vT,vBot=vB)
    xr = np.log(sm)
    xr = xr.reshape(xr.shape[0],1)
    xr_R = xr
    xr_R = xr_R.reshape(xr_R.shape[0],1)


    fob = mgr.createForwardOperator()
    fob.setData(data)
    fob.setMesh(mesh)



    # data weight matrix
    rmag_linear = rhos1
    dobs = rhos1
    dobs = dobs.reshape(dobs.shape[0],1)

    Wd = np.diag(1.0/ np.abs(rmag_linear[:,0] * 0.01))
    WdTwd = Wd.T.dot(Wd)

    # model weighting matrix
    rm = fob.regionManager()
    Ctmp = pg.matrix.RSparseMapMatrix()
    rm.setConstraintType(1)
    rm.fillConstraints(Ctmp)

    Wm_r = pg.utils.sparseMatrix2coo(Ctmp)
    Wm_r = Wm_r.todense()
    Wm_r = L_r*Wm_r
    chi2 = 1
    for i in range(10):
        if flag==1:
            dr,J = inv.srtforandjac2(fob, xr)
        else:
            dr,J = inv.srtforandjac3(fob, xr)


        dr = dr.reshape(dr.shape[0],1)


        grad_fd = np.dot(np.dot(J.transpose(),WdTwd), (dr-dobs))
        gg = grad_fd + np.dot(Wm_r.transpose().dot(Wm_r),xr-xr_R)

        N11 =np.vstack((Wd.dot(J), Wm_r))

        dataerror = dr - dobs
        gc1 = np.vstack((Wd.dot(dataerror),(Wm_r.dot(xr-xr_R))))
        print('Iterations'+str(i+1))
        print('max error:'+str(np.max(dataerror/dobs)))

        fd = (np.dot(Wd, dataerror)).T.dot(np.dot(Wd,dataerror))

        fm = (Wm_r*(xr-xr_R)).T.dot(Wm_r*(xr-xr_R))

        fc = fm + fd

        chi2new = fd/len(dr)
        print('chi2:'+str(chi2new))
        print('dPhi:'+str(abs(chi2-chi2new)/chi2))
        if chi2new < 1.5 or abs(chi2-chi2new)/chi2 < 0.01:
            print('chi2='+str(fd/len(dr)))
            print('dPhi:'+str(abs(chi2-chi2new)/chi2))
            break

        chi2 = chi2new

        sN11 = sparse.csr_matrix(N11)
        mu_LS = 1
        iarm = 1
        gc2 = np.array(gc1)
        gc = gc2.reshape((-1,))

        d, istop, itn, normr = lsqr(sN11, -gc, damp=0.0, atol=1e-08, btol=1e-08, conlim=100000000.0, iter_lim=None, show=False, calc_var=False, x0=None)[:4]
        d = d.reshape(d.shape[0],1)

        while 1:
            xt = xr + mu_LS*d
            if flag==1:
                dr1 = inv.srtforward2(fob, xt)
            else:
                dr1 = inv.srtforward3(fob, xt)



            dr1 = dr1.reshape(dr1.shape[0],1)
            dataerror = dr1 - dobs
            fd = (np.dot(Wd, dataerror)).T.dot(np.dot(Wd,dataerror))
            fm = (Wm_r*(xt-xr_R)).T.dot(Wm_r*(xt-xr_R))
            ft = fm + fd

            fgoal = fc - 1e-4*mu_LS*(d.T.dot(gg.reshape(gg.shape[0],1)))

            #print('ft'+str(ft))
            #print('fgoal'+str(fgoal))
            if ft < fgoal:
                break
            else:
                iarm = iarm+1
                mu_LS = mu_LS/2

            if iarm > 20:
                pg.boxprint('Line search FAIL EXIT')
                break


        xr = xt
        dr = dr1
        xr[xr > np.log(1/50)] = np.log(1/50)
        xr[xr < np.log(1/5000)] = np.log(1/5000)

    coverage = fob.jacobian().transMult(np.ones(fob.data.size()))
    CC = Ctmp
    NN = np.sign(np.absolute(CC.transMult(CC * coverage)))
    return xr,NN


def srtporosity(rhos1, mesh, L_r, data, mgr, flag=1,vT=1/0.7, vB=1/0.3):

    #sm = np.ones((mesh.cellCount()))*0.5
    sm = tt.createGradientModel2D(data,mesh,  vTop=vT,vBot=vB)
    xr = sm
    xr = xr.reshape(xr.shape[0],1)
    xr_R = xr
    xr_R = xr_R.reshape(xr_R.shape[0],1)


    fob = mgr.createForwardOperator()
    fob.setData(data)
    fob.setMesh(mesh)



    # data weight matrix
    rmag_linear = rhos1
    dobs = rhos1
    dobs = dobs.reshape(dobs.shape[0],1)

    Wd = np.diag(1.0/ np.abs(rmag_linear[:,0] * 0.01))
    WdTwd = Wd.T.dot(Wd)

    # model weighting matrix
    rm = fob.regionManager()
    Ctmp = pg.matrix.RSparseMapMatrix()
    rm.setConstraintType(1)
    rm.fillConstraints(Ctmp)

    Wm_r = pg.utils.sparseMatrix2coo(Ctmp)
    Wm_r = Wm_r.todense()
    Wm_r = L_r*Wm_r
    chi2 = 1
    for i in range(10):
        if flag==1:
            dr,J = srtpoJ2(fob, xr)
        else:
            dr,J = srtpoJ3(fob, xr)


        dr = dr.reshape(dr.shape[0],1)


        grad_fd = np.dot(np.dot(J.transpose(),WdTwd), (dr-dobs))
        gg = grad_fd + np.dot(Wm_r.transpose().dot(Wm_r),xr-xr_R)

        N11 =np.vstack((Wd.dot(J), Wm_r))

        dataerror = dr - dobs
        gc1 = np.vstack((Wd.dot(dataerror),(Wm_r.dot(xr-xr_R))))
        print('Iterations'+str(i+1))
        print('max error:'+str(np.max(dataerror/dobs)))

        fd = (np.dot(Wd, dataerror)).T.dot(np.dot(Wd,dataerror))

        fm = (Wm_r*(xr-xr_R)).T.dot(Wm_r*(xr-xr_R))

        fc = fm + fd
        chi2new = fd/len(dr)
        print('chi2:'+str(chi2new))
        print('dPhi:'+str(abs(chi2-chi2new)/chi2))
        if chi2new < 1.5 or abs(chi2-chi2new)/chi2 < 0.01:
            print('chi2='+str(fd/len(dr)))
            print('dPhi:'+str(abs(chi2-chi2new)/chi2))
            break
        chi2 = chi2new
        sN11 = sparse.csr_matrix(N11)
        mu_LS = 1
        iarm = 1
        gc2 = np.array(gc1)
        gc = gc2.reshape((-1,))

        d, istop, itn, normr = lsqr(sN11, -gc, damp=0.0, atol=1e-08, btol=1e-08, conlim=100000000.0, iter_lim=None, show=False, calc_var=False, x0=None)[:4]
        d = d.reshape(d.shape[0],1)

        while 1:
            xt = xr + mu_LS*d
            if flag == 1:
                dr1 = srtpoF2(fob, xt)
            else:
                dr1 = srtpoF3(fob, xt)

            dr1 = dr1.reshape(dr1.shape[0],1)
            dataerror = dr1 - dobs
            fd = (np.dot(Wd, dataerror)).T.dot(np.dot(Wd,dataerror))
            fm = (Wm_r*(xt-xr_R)).T.dot(Wm_r*(xt-xr_R))
            ft = fm + fd

            fgoal = fc - 1e-4*mu_LS*(d.T.dot(gg.reshape(gg.shape[0],1)))

            #print('ft'+str(ft))
            #print('fgoal'+str(fgoal))
            if ft < fgoal:
                break
            else:
                iarm = iarm+1
                mu_LS = mu_LS/2

            if iarm > 20:
                pg.boxprint('Line search FAIL EXIT')
                break


        xr = xt
        dr = dr1
        xr[xr > 1] = 1
        xr[xr < 0.01] = 0.01



    J3 = srtpoJ(fob,xr)
    coverage = J3.transMult(np.ones(fob.data.size()))
    CC = Ctmp
    NN = np.sign(np.absolute(CC.transMult(CC * coverage)))
    return xr,NN



def jointporosity(rhos1, mesh, grid, L_r, dataert, datasrt, ert, mgr,flag=1 ,vT=1/0.7, vB=1/0.3):

    sm = tt.createGradientModel2D(datasrt, mesh,  vTop=vT, vBot=vB)
    xr = sm
    xr = xr.reshape(xr.shape[0],1)
    xr_R = xr
    xr_R = xr_R.reshape(xr_R.shape[0],1)


    fobert = ert.ERTModelling()
    fobert.setData(dataert)
    fobert.setMesh(grid)

    fobsrt = mgr.createForwardOperator()
    fobsrt.setData(datasrt)
    fobsrt.setMesh(mesh)

    # data weight matrix
    rmag_linear = rhos1
    dobs = rhos1
    dobs = dobs.reshape(dobs.shape[0],1)

    Wd = np.diag(1.0/ np.abs(rmag_linear[:,0] * 0.01))
    WdTwd = Wd.T.dot(Wd)

    # model weighting matrix
    rm = fobert.regionManager()
    Ctmp = pg.matrix.RSparseMapMatrix()
    rm.setConstraintType(1)
    rm.fillConstraints(Ctmp)

    Wm_r = pg.utils.sparseMatrix2coo(Ctmp)
    Wm_r = Wm_r.todense()
    Wm_r = L_r*Wm_r
    chi2 = 1
    for i in range(10):
        if flag==1:
            dr,J = jointpoJ3(fobert,fobsrt,xr,mesh)
        else:
            dr,J = jointpoJ2(fobert,fobsrt,xr,mesh)



        grad_fd = np.dot(np.dot(J.transpose(),WdTwd), (dr-dobs))
        gg = grad_fd + np.dot(Wm_r.transpose().dot(Wm_r),xr-xr_R)

        N11 =np.vstack((Wd.dot(J), Wm_r))

        dataerror = dr - dobs
        gc1 = np.vstack((Wd.dot(dataerror),(Wm_r.dot(xr-xr_R))))
        print('Iterations'+str(i+1))
        print('max error:'+str(np.max(dataerror/dobs)))

        fd = (np.dot(Wd, dataerror)).T.dot(np.dot(Wd,dataerror))

        fm = (Wm_r*(xr-xr_R)).T.dot(Wm_r*(xr-xr_R))

        fc = fm + fd

        chi2new = fd/len(dr)
        print('chi2:'+str(chi2new))
        print('dPhi:'+str(abs(chi2-chi2new)/chi2))
        if chi2new < 1.5 or abs(chi2-chi2new)/chi2 < 0.01:
            print('chi2='+str(fd/len(dr)))
            print('dPhi:'+str(abs(chi2-chi2new)/chi2))
            break

        chi2 = chi2new

        sN11 = sparse.csr_matrix(N11)
        mu_LS = 1
        iarm = 1
        gc2 = np.array(gc1)
        gc = gc2.reshape((-1,))

        d, istop, itn, normr = lsqr(sN11, -gc, damp=0.0, atol=1e-08, btol=1e-08, conlim=100000000.0, iter_lim=None, show=False, calc_var=False, x0=None)[:4]
        d = d.reshape(d.shape[0],1)

        while 1:
            xt = xr + mu_LS*d

            if flag == 1:
                dr1 = jointpoF3(fobert,fobsrt,xt,mesh)
            else:
                dr1 = jointpoF2(fobert,fobsrt,xt,mesh)


            dataerror = dr1 - dobs
            fd = (np.dot(Wd, dataerror)).T.dot(np.dot(Wd,dataerror))
            fm = (Wm_r*(xt-xr_R)).T.dot(Wm_r*(xt-xr_R))
            ft = fm + fd

            fgoal = fc - 1e-4*mu_LS*(d.T.dot(gg.reshape(gg.shape[0],1)))

            #print('ft'+str(ft))
            #print('fgoal'+str(fgoal))
            if ft < fgoal:
                break
            else:
                iarm = iarm+1
                mu_LS = mu_LS/2

            if iarm > 20:
                pg.boxprint('Line search FAIL EXIT')
                break

        xt[xt > 1] = 1
        xt[xt < 0.01] = 0.01
        xr = xt
        dr = dr1


    return xr
