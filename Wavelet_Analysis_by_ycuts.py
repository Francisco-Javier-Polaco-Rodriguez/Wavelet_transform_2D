#from re import S
import numpy as np
#import matplotlib.pyplot as plt
import cwt_2dV2 as wv
from scipy.io import savemat,loadmat
import shutil
import os

def t_to_index(t,tmin,tmax,dt = 0.18):
    if tmin < t[0]:
        raise ValueError("t_in is smaller than t[0]=%1.2f, so is not in the t domain in simulation"%(t[0]))
    elif tmax > t[-1]:
        print('tmin =', tmin)
        print('tmax =', tmax)
        print('t = ', t)
        
        raise ValueError("t_fin is biger than t[-1]=%1.2f, so is not in the t domain in simulation"%(t[-1]))
    index_tmin = int((tmin-t[0])/dt)
    index_tmax = int((tmax-t[0])/dt)
    return index_tmin,index_tmax

def coef(Dw,dt,dx,ncores = 4,precission = 1):
    [x,t] = np.meshgrid(np.arange(0,423*0.01/Dw[1],dx),np.arange(0,54*0.1/Dw[0],dt))
    [w,k] = np.meshgrid(np.linspace(1-Dw[0],1+Dw[0],30*precission),np.linspace(0.1-Dw[1],0.1+Dw[1],30*precission))
    A = lambda x,t : 5*np.sin(0.1*x-1*t)
    Axt = A(x,t)
    C = wv.cwt_2d(Axt,w,k,dt,dx,Dw,ncores = ncores)
    cft = 0.5*25/np.max(C[np.int32(Axt.shape[0]/2),:,:])
    del x,t,w,k,C
    return cft

def savemat_in_parts(cft,coefficient,t,w,k,N_mat,folder,name):
    dir = os.path.join(folder)
    if not os.path.exists(dir):
        os.mkdir(dir)
    else:
        print('\nWARNING: you are saveing in an existant directory. Your results may be mixed with whatever you have in %s'%(dir))

    N_in_each_mat = np.int32(len(t)/N_mat)
    for s in range(N_mat):
        if s+1 != N_mat:
            dic = {'Normalization_coeficient':coefficient,'Sum_cft_square':cft[s*N_in_each_mat:(s+1)*N_in_each_mat,:,:],'w':w,'k':k,'t':t[s*N_in_each_mat:(s+1)*N_in_each_mat]}
            savemat(folder+'/'+ 'Part%iof%i_'%(s+1,N_mat) + name +'.mat',dic)
        else:
            dic = {'Normalization_coeficient':coefficient,'Sum_cft_square':cft[s*N_in_each_mat:-1,:,:],'w':w,'k':k,'t':t[s*N_in_each_mat:-1]}
            savemat(folder+'/'+ 'Part%iof%i_'%(s+1,N_mat) + name +'.mat',dic)


def doall(Path,niDn0005, wmin,wmax,kmin,kmax,N_points_w,N_points_k,Dw,Tmin,Tmax,N_mat,field,Dn,folder_to_write,coefficient,dt=0.18,dx=1.41,ncores = 4,DT = 1):
    print('Reading %s for Dn = %1.2f'%(field,Dn))
    dt,t,dx,x,Axt = wv.read(Path,niDn0005,N='all',dt=dt,dx=dx)
    indx = np.arange(0,len(t),DT)
    Axt = Axt[indx,:]
    t = t[indx]
    dt = DT*dt
    indx_min,indx_max = t_to_index(t,Tmin,Tmax,dt = dt)
    t = t[indx_min:indx_max]
    Axt = Axt[indx_min:indx_max,:]
    [w,k] = np.meshgrid(np.linspace(wmin,wmax,N_points_w),np.linspace(kmin,kmax,N_points_k),indexing = 'ij')
    print('\nCalculating my version of continuous wavelet transform.\nAxt.shape = %ix%i'%(Axt.shape[0],Axt.shape[1]))
    if Dw[0]<5/np.abs(t[0]-t[-1]) or Dw[1]<5/np.abs(x[0]-x[-1]):
        raise ValueError('Dk and Dw too low. Yo may never go under 5/DT for Dw or 5/Dx dor k')
    cft = wv.cwt_2d(Axt,w,k,dt,dx,Dw,ncores = ncores)
    name = '%s_with_Dn=%1.3f_analysis_for_wmin=%1.3f_wmax=%1.3f_kmin=%1.3f_kmax=%1.3f_Dw=%1.3f_Dk=%1.3f__Between_T=%1.0f_T=%1.0f'%(field,Dn,wmin,wmax,kmin,kmax,Dw[0],Dw[1],t[0],t[-1])
    savemat_in_parts(cft*coefficient,coefficient,t,w,k,N_mat,folder_to_write,name)
    del dt,t,dx,x,Axt,cft,w,k
    print('\n')



# yvec = [64,128,192]#,256,320,384,448,512,576,640,704,768,832,896,960,1022]


# ########## ni_tranches
# Path_to_read = '/home/polanco/Ondlettes/Par_tranches/ni_Dn=0.05_T=0_T=end'
# Path_to_write = '/run/media/polanco/BOLET-Xusb3/Ondlettes'
# name_of_folder_to_save_mat = 'n_i_Dn=0.005_16_tranches_alltime_test'

# niDn0005=True

# wmin = 0
# wmax = 0.08
# kmin = -0.3
# kmax = 0.3
# N_points_w = 100
# N_points_k = 100

# Dk = 0.005
# Dw = 0.005

# dt = 1.8
# dx = 1.41

# Tmin = 0
# Tmax = 5000

# Number_of_times_skip = 1
# N_mat = 10

# field = 'n_i'
# Dn = 0.05

# ncores = 16

def waveletAnalysis(parametresWavelet):

    Path_to_read = parametresWavelet['PathToRead']
    Path_to_write = parametresWavelet['PathToWrite']

    wmin = parametresWavelet['omega_min']
    wmax = parametresWavelet['omega_max']
    kmin = parametresWavelet['k_min']
    kmax = parametresWavelet['k_max']
    N_points_w = parametresWavelet['N_points_w']
    N_points_k = parametresWavelet['N_points_k']
    Dw = parametresWavelet['Delta_omega']
    Dk = parametresWavelet['Delta_k']

    dt = parametresWavelet['Dt']
    dx = parametresWavelet['dx']
    Tmin = parametresWavelet['T_init']
    Tmax = parametresWavelet['T_end']
    Number_of_times_skip = parametresWavelet['Number_of_times_skip']
    N_mat = parametresWavelet['N_mat']
#    print(N_mat)
    Dn= parametresWavelet['fluctuations']
    ncores= parametresWavelet['Ncores']
    yvec= parametresWavelet['ycut']
    
    if ('filtered' in Path_to_read):
        field=parametresWavelet['field']+'_filtered'
    else:
        field=parametresWavelet['field']
        
    name_of_folder_to_save_mat = 'MAT_'+os.path.basename(Path_to_read)

    try:
        os.mkdir(Path_to_write +os.sep+name_of_folder_to_save_mat)
    except FileExistsError:
        yn = input('WARNING: existant directory. Do you want to remove the old one? (y/n)')
        bool = yn.capitalize() == 'Y'
        if bool:
            shutil.rmtree(Path_to_write+os.sep+name_of_folder_to_save_mat)
            os.mkdir(Path_to_write+os.sep+name_of_folder_to_save_mat)
        else:
            exit("Ending program. Change the name of the folder in 'name_of_folder_to_save_mat' in Analysis_by_ycuts.py manually and restart.")
    
    for i,y in enumerate(yvec):
        Name_of_ycut_folder = '%s_Dn=%1.2f_Y=%i'%(field,Dn,y)
    
        name_mat = '%s_at_Y=%i_with_Dn=%1.3f_data_for_wmin=%1.3f_wmax=%1.3f_kmin=%1.3f_kmax=%1.3f_Dw=%1.3f_Dk=%1.3f_T=%1.2f_T=%1.2f'%(field,y,Dn,wmin,wmax,kmin,kmax,Dw,Dk,Tmin,Tmax)
    
        if i == 0:
            print('Calculating proportionality coefficient for Dw=%1.4f and Dk = %1.4f\n'%(Dw,Dk))
            coefficient = coef(np.array([Dw,Dk]),dt,dx,ncores = ncores)
            print('\nCoefficient = %f'%(coefficient))
        print('For Y = %i LD\n'%(y))

        doall(Path_to_read+os.sep+Name_of_ycut_folder,parametresWavelet['nomFichierMat'], wmin,wmax,kmin,kmax,
              N_points_w,N_points_k,np.array([Dw,Dk]),Tmin,Tmax,N_mat,field,Dn,
              Path_to_write+'/'+name_of_folder_to_save_mat+'/'+name_mat,coefficient,
              ncores = ncores,DT=Number_of_times_skip,dt=dt,dx=dx)
    
    print('\nNow summing all the .mat generated')
    
    for n,y in enumerate(yvec):
        path = Path_to_write+'/'+name_of_folder_to_save_mat+'/'+'%s_at_Y=%i_with_Dn=%1.3f_data_for_wmin=%1.3f_wmax=%1.3f_kmin=%1.3f_kmax=%1.3f_Dw=%1.3f_Dk=%1.3f_T=%1.2f_T=%1.2f'%(field,y,Dn,wmin,wmax,kmin,kmax,Dw,Dk,Tmin,Tmax)
        print('\ny = %i\nLoading....'%(y))
    
        for roots, directorys, files in os.walk(path):
                files = np.sort(files)
                for i,file in enumerate(files):
                    print('     '+file)
                    data = loadmat(path + '/'+file)
                    if i == 0:
                        k = data['k']
                        w = data['w']
                        Sum_x_cft_square = data['Sum_cft_square']
                        t = data['t'][0]
                        coefficient = data['Normalization_coeficient']
                    else:
                        Sum_x_cft_square = np.concatenate((Sum_x_cft_square,data['Sum_cft_square']),axis = 0)
                        t = np.concatenate((t,data['t'][0]))
                if n == 0:
                    Sum_xy_cft_square = Sum_x_cft_square
                else:
                    Sum_xy_cft_square = Sum_xy_cft_square + Sum_x_cft_square
    
    pathTocreateCoeffSummation = Path_to_write+'/'+'SUM_OF_COEFFICIENTS_%s_with_Dn=%1.3f_data_for_wmin=%1.3f_wmax=%1.3f_kmin=%1.3f_kmax=%1.3f_Dw=%1.3f_Dk=%1.3f_T=%1.2f_T=%1.2f'%(field,Dn,wmin,wmax,kmin,kmax,Dw,Dk,Tmin,Tmax)
    savemat_in_parts(Sum_xy_cft_square,coefficient,t,w,k,N_mat, pathTocreateCoeffSummation,'SUM_OF_COEFFICIENTS_%s_with_Dn=%1.3f_data_for_wmin=%1.3f_wmax=%1.3f_kmin=%1.3f_kmax=%1.3f_Dw=%1.3f_Dk=%1.3f_T=%1.2f_T=%1.2f'%(field,Dn,wmin,wmax,kmin,kmax,Dw,Dk,Tmin,Tmax))
    return pathTocreateCoeffSummation

