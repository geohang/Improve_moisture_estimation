import numpy as np
import pygimli as pg
import pygimli.physics.petro as petro
from scipy.optimize import fsolve

class petrorange:
    def __init__(self):
        self.porosity = []
        self.Sat = []
        self.m_model = []
        self.n_model = []
        self.a_model = []
        self.rFluid = []
        self.sigmas = []
        self.rhos = []

class petroclass:
    def __init__(self):
        self.porosity = []
        self.Sat = []
        self.m_model = []
        self.n_model = []
        self.a_model = []
        self.rFluid = []
        self.sigmas = []
        self.rhos = []
 

## using rhos
def Archierhos(Sat=1,rhos=100, n=2, a=1,sigma_sur=0):
    sigma_sat= 1/rhos# S(rho)
    sigma = sigma_sat*Sat**n + sigma_sur*Sat**(n-1)
    res = 1/sigma
    return  res
    
def InvArchieSrhos(rho,rhos=100, n=2, a=1,sigmas=0):  # S(rho)
    """Inverse Archie transformation function saturation(resistivity)."""
    N = (a*rhos/rho)
    return 1/(1/N+sigmas)**(1/n)


def InvArchieSrhos2(rho,rhos=100, n=2, a=1,sigma_sur=0):  # S(rho)
    """Inverse Archie transformation function saturation(resistivity)."""
    sigma_sat = 1/rhos
    S_2_rho = InvArchieSrhos(rho=rho,rhos = rhos, n = n, a=1)
    
    solution = []
    for i in range(len(sigma_sur)):
        if sigma_sur[i]==0:
            solution.append(S_2_rho[i])
        else:
            func = lambda SS : sigma_sat[i]*SS**n[i] + sigma_sur[i]*SS**(n[i]-1) - 1/rho[i]
            solution.append(fsolve(func, S_2_rho[i]))
            
    solution = np.array(solution)            
    return solution

    
def Archie(Sat=1,rFluid=20, phi=0.4, m=2, n=2, a=1,sigmas=0):  # S(rho)
    N = a*rFluid*phi**(-m)*Sat**(-n)
    return  1/(1/N+sigmas)
    
    
def waterArchie(WC=1,rFluid=20, phi=0.4, m=2, n=2, a=1,sigmas=0):  # S(rho)
    N = a*rFluid*phi**(-m+n)*WC**(-n)
    return  1/(1/N+sigmas)

    
def InvArchieS(rho,rFluid=20, phi=0.4, m=2, n=2, a=1,sigmas=0):  # S(rho)
    """Inverse Archie transformation function saturation(resistivity)."""
    N = (a*rFluid*phi**(-m)/rho)
    return 1/(1/N+sigmas)**(1/n)
    
def waterInvArchieS(rho,rFluid=20, phi=0.4, m=2, n=2, a=1,sigmas=0): 
    """Inverse Archie transformation function saturation(resistivity)."""
    N = (a*rFluid*phi**(-m+n)/rho)
    return 1/(1/N+sigmas)**(1/n)
    

def ArchieSderi(a=1, n=2, rFluid=20, phi=0.3,m=2, S=0.7,sigmas=0):
    N = a*rFluid*phi**(-m)*S**(-n)
    
    return 1/((1+sigmas*N)**2)*(-n*a*rFluid/(phi**(m)*S**(n+1)))
    
    
def waterArchieSderi(a=1, n=2, rFluid=20, phi=0.3,m=2, WC=0.2,sigmas=0):
    N = a*rFluid*phi**(-m+n)*WC**(-n)
    
    return 1/((1+sigmas*N)**2)*(-n*a*rFluid/(phi**(m-n)*WC**(n+1)))
    
    
def proassign(markerall2,pro1,pro2,pro3):
    property1 = np.zeros((markerall2.shape))
    property1[markerall2==3] = pro1
    property1[markerall2==0] = pro2
    property1[markerall2==2] = pro3
    return property1



class PetroModelling2(pg.frameworks.MeshModelling):
    """Combine petrophysical relation with the modelling class f(p).

    Combine petrophysical relation :math:`p(m)` with a modelling class
    :math:`f(p)` to invert for the petrophysical model :math:`p` instead
    of the geophysical model :math:`m`.

    :math:`p` be the petrophysical model, e.g., porosity, saturation, ...
    :math:`m` be the geophysical model, e.g., slowness, resistivity, ...

    """
    def __init__(self, fop, petro1, **kwargs):
        """Save forward class and transformation, create Jacobian matrix."""
        self._f = fop
        # self._f createStartModel might be called and depends on the regionMgr
        self._f.regionManager = self.regionManager
        # self.createRefinedFwdMesh depends on the refinement strategy of self._f
        # self.createRefinedFwdMesh = self._f.createRefinedFwdMesh

        super(PetroModelling2, self).__init__(fop=None, **kwargs)
        # petroTrans.fwd(): p(m), petroTrans.inv(): m(p)
        self._petroTrans = petro1  # class defining p(m)

        self._jac = pg.matrix.MultRightMatrix(self._f.jacobian())
        self.setJacobian(self._jac)



    def setMeshPost(self, mesh):
        """ """
        self._f.setMesh(mesh, ignoreRegionManager=True)


    def setDataPost(self, data):
        """ """
        self._f.setData(data)


    def createStartModel(self, data):
        """Use inverse transformation to get m(p) for the starting model."""
        sm = self._f.createStartModel(data)
        pModel = waterInvArchieS(rho=sm,rFluid=self._petroTrans.rFluid, phi=self._petroTrans.porosity, 
                            m=self._petroTrans.m_model, n=self._petroTrans.n_model, a=self._petroTrans.a_model,sigmas=self._petroTrans.sigmas)
        pModel[pModel>0.5] = 0.5
        pModel[pModel<0.05] = 0.05
        pModel = pg.matrix.RVector(pModel)
        return pModel


    def response(self, model):
        """Use transformation to get p(m) and compute response f(p)."""
        #tModel = petro.resistivityArchie(rFluid=self._petroTrans.rFluid, porosity=self._petroTrans.porosity,
        #                                 a=self._petroTrans.a_model, m=self._petroTrans.m_model, sat=model, n=self._petroTrans.n_model)
        tModel = waterArchie(WC=model,rFluid=self._petroTrans.rFluid, phi=self._petroTrans.porosity,
                m=self._petroTrans.m_model, n=self._petroTrans.n_model, a=self._petroTrans.a_model,sigmas=self._petroTrans.sigmas)
        ret = self._f.response(tModel)
        return ret


    def createJacobian(self, model):
        r"""Fill the individual jacobian matrices.
        J = dF(m) / dm = dF(m) / dp  * dp / dm
        """
        #tModel = petro.resistivityArchie(rFluid=self._petroTrans.rFluid, porosity=self._petroTrans.porosity,
        #                                 a=self._petroTrans.a_model, m=self._petroTrans.m_model, sat=model, n=self._petroTrans.n_model)
        tModel = waterArchie(WC=model,rFluid=self._petroTrans.rFluid, phi=self._petroTrans.porosity,
                        m=self._petroTrans.m_model,n=self._petroTrans.n_model, a=self._petroTrans.a_model,sigmas=self._petroTrans.sigmas)
        self._f.createJacobian(tModel)
        self._jac.A = self._f.jacobian()

        r = waterArchieSderi(a=self._petroTrans.a_model, n=self._petroTrans.n_model, 
                           rFluid=self._petroTrans.rFluid, phi=self._petroTrans.porosity,m=self._petroTrans.m_model, WC=model,sigmas=self._petroTrans.sigmas)
        r = pg.matrix.RVector(r)
        self._jac.r = r  # set inner derivative

        # print(self._jac.A.rows(), self._jac.A.cols())
        # print(self._jac.r)
        # pg._r("create Jacobian", self, self._jac)
        self.setJacobian(self._jac) # to be sure .. test if necessary


class PetroInversionManager2(pg.frameworks.MeshMethodManager):
    """Class for petrophysical inversion (s. Rücker et al. 2017)."""
    def __init__(self, petro, mgr=None, **kwargs):
        """Initialize instance with manager and petrophysical relation."""
        petrofop = kwargs.pop('petrofop', None)
        #print(petrofop)
        if petrofop is None:
            fop = kwargs.pop('fop', None)
            #print(fop)
            if fop is None and mgr is not None:
                # Check! why I can't use mgr.fop
                #fop = mgr.fop
                fop = mgr.createForwardOperator()
                self.checkData = mgr.checkData
                self.checkError = mgr.checkError

            if fop is not None:
                if not isinstance(fop, PetroModelling2):
                    petrofop = PetroModelling2(fop, petro)

        if petrofop is None:
            print(mgr)
            print(fop)
            pg.critical('implement me')

        super().__init__(fop=petrofop, **kwargs)


