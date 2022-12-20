### Coded by: Francisco Javier Polanco Rodríguez, working to Institut polytechnique de Paris.
#Graduated in Universidad de Sevilla, Spain.

#Importing modules--------------------------------------------------------

import numpy as np;                 import matplotlib.pyplot as plt
import pywt;                        from scipy.io import loadmat,savemat
import imageio;                     import os
from progress.bar import Bar;       from threading import Thread
from functools import lru_cache;    import pyfftw.interfaces.scipy_fftpack as fft

## Optimizating the pyfftw.interfaces by letting acces to de cache, since several fft are done in a cwt.

import pyfftw
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(10)

###---Previous-definitions-------------------------------------------------
sin = np.sin
cos = np.cos
pi = np.pi
# Thread doesn't returns the return value of the target function. It is necesary to create a class that heritates Thread and witch returns the value. Be carefull it is donde for Python3. If you don't use python3, delete this class and uncoment :
#############################################################################################

#class ThreadWithReturnValue(Thread):
#    def __init__(self, group=None, target=None, name=None,
#                args=(), kwargs={}, Verbose=None):
#        Thread.__init__(self, group, target, name, args, kwargs, Verbose)
#        self._return = None
#    def run(self):
#        if self._Thread__target is not None:
#            self._return = self._Thread__target(*self._Thread__args,
#                                                **self._Thread__kwargs)
#    def join(self):
#        Thread.join(self)
#        return self._return

############################################################################################

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        #print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

def morlet(w,k,w0 = np.array([6,0]),Dw = np.array([1,1])):
    return np.exp(-(w-w0[0])**2/2/Dw[0]**2)*np.exp(-(k+w0[1])**2/2/Dw[1]**2)

@lru_cache(5) # It makes computer search in cache memory usefull data
def _create_frequency_plane(Axt_shape: tuple):
    """
    It makes the frequency plane at the fft space. We use that to evaluate the fft of the wavelet before doing the ifft2.
    """
    assert len(Axt_shape) == 2

    h, w = Axt_shape
    w_2 = (w - 1) // 2
    h_2 = (h - 1) // 2

    w_pulse = 2 * np.pi / w * np.hstack((np.arange(0, w_2 + 1), np.arange(w_2 - w + 1, 0)))
    h_pulse = 2 * np.pi / h * np.hstack((np.arange(0, h_2 + 1), np.arange(h_2 - h + 1, 0)))

    xx, yy = np.meshgrid(w_pulse, h_pulse, indexing='xy')
    dxx_dyy = abs((xx[0, 1] - xx[0, 0]) * (yy[1, 0] - yy[0, 0]))

    return xx, yy

def target_cwt2d(fft2Axt,w0,w,k,Domega): # Old
    """
    Wavelet transform for only one (w_i,k_i), we use this fonction like target for doing threading.
    """
    if (np.abs(w0[0]/Domega[0])**2+np.abs(w0[1]/Domega[1])**2)**0.5 < 6:
        Sum_cft_x = np.full(fft2Axt[:,0].shape,np.nan)
    else:
        Sum_cft_x = np.sum(np.abs(fft.ifft2(fft2Axt*morlet(w,k,w0 = w0,Dw = Domega)))**2,axis = 1) #Sum_x{IFFT2[FFT2(Axt)*psi]²}(t)
    return Sum_cft_x


def cwt_2d(Axt,w_in,k_in,dt,dx,Domega,ncores=1):
    """
    Variation of the continuous 2d wavelet transform. It will analyce the input fonction Axt(x,t) in space and frequency. The autput is Sum_x(|W(x,t,k,f)|^2) for not using too much RAM memory, and because our interest is to calculate w(t) and k(t).
    Inputs :
    -Axt : the nxm matrix to do the transform
    -w and k : they must to be 2 matrices of the same shape with w[i,j] and k[i,j] representing the couple (w,k)_ij to analyce. Usually what we call a meshgrid.
        EXEMPLE:
        # If we want kmin = -0.3, kmax = 0.3, wmin = 0, wmax = 3, we do
        [w,k] = np.meshgrid(np.linspace(0,3,100),np.linspace(-0.3,0.3,100),indexing = 'ij')
    -dt,dx : the sample time of the temporal signal and the spatial distance in the spatial signal
    -Domega : a vector representing the standard desviation of the gaussian wavelet filter in the Fourier space for w and for k. 

        NOTE: all the values w,k,dw,dk,dt,dx are physicals. The program adimentionalice with all this information.
    -ncores : optional, if you have a powerfull setup (or at least more than 1 core) you can paralelyce multiple transforms. ncores represents the number of process that are going to be done at the same time.
    """
    if k_in.shape != w_in.shape:
        raise ValueError("wx and wy imputs doesn't match")
    shap = w_in.shape
    Domega = Domega*np.array([dt,dx])
    Sum_cft_x = np.zeros([Axt.shape[0],shap[0],shap[1]],dtype='float32')
    k,w = _create_frequency_plane(Axt.shape)
    fft2Axt = fft.fft2(Axt)

    # We remember the better way of calculating a fft with such dimmentions.

    wisdom = pyfftw.export_wisdom()
    pyfftw.import_wisdom(wisdom)

    bar = Bar('Processing', max = shap[0]*shap[1],suffix='%(percent)d%%(%(elapsed)ds elapsed) %(eta)ds remaining')
    for i in range(shap[0]):
        for s in range(np.int32(shap[1]/ncores)):
            if s != np.int32(shap[1]/ncores)-1:
                th = []
                J = np.arange(s*ncores,(s+1)*ncores,1)
                for j in J:
                    w0 = np.array([w_in[i,j]*dt,k_in[i,j]*dx])
                    th.append(ThreadWithReturnValue(target = target_cwt2d,args = (fft2Axt,w0,w,k,Domega)))
                for thread in th:
                    thread.start()
                for j,thread in enumerate(th):
                    Sum_cft_x[:,i,J[j]] = thread.join()
                    bar.next()
            else:
                th = []
                J = np.arange(s*ncores,shap[1],1)
                for j in J:
                    w0 = np.array([w_in[i,j]*dt,k_in[i,j]*dx])
                    th.append(ThreadWithReturnValue(target = target_cwt2d,args = (fft2Axt,w0,w,k,Domega)))
                for thread in th:
                    thread.start()
                for j,thread in enumerate(th):
                    Sum_cft_x[:,i,J[j]] = thread.join()
                    bar.next()
    return Sum_cft_x

def read(path,niDn0005,N = 'all' ,dt = 0.18,dx = 1.41,N_v = 1024):
    """
    Dis fonction takes the path of the directory containing ONLY .mat. In each .mat we have the profile of a physic magnitude for a time T and all x. Them it returns the sample time (0.18 by default, if you want to change you have to doi manually) the separation of the grid in x dx (1.41 by default, if yu want to change you must do manually) the x and the t of the signal domain Axt(x,t) and the signal Axt
    -path relativepath to the directory taht we want to read.
    -N number of .mat that we want to read. It is 'all' by default, and it read all the .mat in the directory
    -dt sample time, 0.18 by default. 
    -dx distance between squares in the greed of simmulation, 1.41 by default.
    -N_v number of squares in the grid, 1024 by default. If you read directorys with 4028 you have to change this parameter
    """
    keysort = lambda name : float(name[name.find('T=')+2:name.find('.mat')])

    if N == 'all':

        for root, directories, files in os.walk(path,topdown=True):
            files.sort(key = keysort)
            N = len(files)
            T = np.zeros(N)
            Axt = np.zeros([N,N_v])
            bar = Bar('Reading', max = N,suffix='%(percent)d%%(%(elapsed)ds elapsed)')
            for i,name in enumerate(files):
                T[i] = float(name[name.find('T=')+2:name.find('.mat')])/100
                data = loadmat(path+'/'+name)
                # if niDn0005:
                #     Axt[i,:] = data['filtered_hp_2']
                # else:
                Axt[i,:] = data[niDn0005]
                bar.next()
            x = data['xplot']
            x = x[0,:]
    else:
        T = np.zeros(N+1)
        Axt = np.zeros([N+1,N_v])
        for root, directories, files in os.walk(path,topdown=True):
            files.sort(key = keysort)
            bar = Bar('Reading', max = N,suffix='%(percent)d%%(%(elapsed)ds elapsed)')
            for i,name in enumerate(files):
                T[i] = float(name[name.find('T=')+2:name.find('.mat')])/100
                data = loadmat(path+'/'+name)
                # if niDn0005:
                #     Axt[i,:] = data['filtered_hp_2']
                # else:
                Axt[i,:] = data[niDn0005]
                bar.next()
                if i == N:
                    break
            x = data['xplot']
            x = x[0,:]
    return dt,T,dx,x,Axt


## -----Thank you for trusting me, I hope my code works well for you :D------
print('--------------------------------\nYou imported my continuous\nwavelet transform module\nThank you for trusting me ;D\n--------------------------------\n   By:\nFrancisco Javier\nPolanco Rodríguez-28/06/2022\n')