import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat,loadmat
import os, glob
import imageio
from progress.bar import Bar
from tqdm import tqdm

from DefaultInputData import LoadModuleInputData
inputData = LoadModuleInputData()


def dofigwtkt(cft,w,k,t,sumy = True,sumybad = False,field = '',Dn = 0,log = True,borne_inf = -3,borne_sup = 3,axiskt = None, axiswt = None,grid = True,logw = False):
    cft[np.isnan(cft)] = 0
    if field == '':
        field = input('What field is it ?')
    wvec = w[:,0]
    kvec = k[0,:]
    wt = np.sum(cft,axis = 2).T#*np.abs(kvec[0]-kvec[1])
    kt = np.sum(cft,axis = 1).T#*np.abs(wvec[0]-wvec[1])
    
    if log:
        wt = np.log10(wt)
        msk = wt <= borne_inf
        wt[msk] = borne_inf
        msk = wt >= borne_sup
        wt[msk] = borne_sup
        kt = np.log10(kt)
        msk = kt <= borne_inf
        kt[msk] = borne_inf
        msk = kt >= borne_sup
        kt[msk] = borne_sup
    fig,ax = plt.subplots(2,1,figsize=(15,20))
    if axiskt != None:
        ax[1].axis(axiskt)
    if axiswt != None:
        ax[0].axis(axiswt)
    ax[0].pcolor(t,wvec,wt,cmap = 'jet')
    plt.grid()
    colors = ax[1].pcolor(t,kvec,kt,cmap = 'jet')
    col = fig.colorbar(colors,orientation = 'horizontal')
    ax[0].set_ylabel('$\omega/\omega_p$',size = 15)
    ax[1].set_ylabel('$k_x\lambda_D$',size = 15)
    ax[1].set_xlabel('$t\omega_p$',size = 15)
    if grid:
        ax[0].grid(b=True, which='major', color='w',alpha = 0.6,linestyle = '-',linewidth = 2);
        ax[0].minorticks_on()
        ax[0].grid(b=True, which='minor', color='w',alpha = 0.4,linestyle = '--',linewidth = 0.9);
        ax[1].grid(b=True, which='major', color='w',alpha = 0.6,linestyle = '-',linewidth = 2);
        ax[1].minorticks_on()
        ax[1].grid(b=True, which='minor', color='w',alpha = 0.4,linestyle = '--',linewidth = 0.9);
    if sumy:
        if sumybad:
            ax[0].set_title(r'$\sum_{\omega}\;\sum_{\forall x}\left|\sum_{\forall y}W_{\omega,k_x}(x,y)\right|^2$ and $\sum_{k_x}\;\sum_{\forall x}\left|\sum_{\forall y}W_{\omega,k_x}(x,y)\right|^2$ for $%s$ and $\Delta n=$%1.2f'%(field,Dn))
        else:
            ax[0].set_title(r'$\sum_{\omega}\;\sum_{\forall x,y}\left|W_{\omega,k_x}(x,y)\right|^2$ and $\sum_{k_x}\;\sum_{\forall x,y}\left|W_{\omega,k_x}(x,y)\right|^2$ for $%s$ and $\Delta n=$%1.2f'%(field,Dn))
            
    if log:
        col.ax.set_xlabel(r'$log_{10}$of energy of all the modes all the modes with $k_x$ or $\omega$',size = 13)
    else:
        col.ax.set_xlabel(r'Energy of all the modes all the modes with $k_x$ or $\omega$',size = 13)
    if logw:
        ax[0].set_yscale('log')
    return fig

def dofig(cft,w,k,Nt,t,axis = None,sumy = True,sumybad = False,field = '',Dn = 0,log = True,borne_inf = -3,borne_sup = 3,grid = True,ytranche = 748):
    if field == '':
        field = input('What field is it ?')
    cft_c = cft[Nt,:,:].copy()
    
    # cft_c = np.log10(cft_c)
    # print('min cft_c = ', np.min(cft_c))
    # print('max cft_c = ', np.max(cft_c))
    # cft_c[np.logical_not(np.isfinite(cft_c))]=0.
    # print('max cft_c = ', np.max(cft_c))
    # print('min cft_c = ', np.min(cft_c))
    # print('k min = ', np.min(k))
    # print('k max = ', np.max(k))
    
    # print('w min = ', np.min(w))
    # print('w max = ', np.max(w))
    # print(np.shape(cft_c))
    # print(np.shape(k))
    # print(np.shape(w))
    # print(w[:,1])
    print('borne_inf = ', borne_inf)
    print('borne_sup = ', borne_sup)
    
#    cft_c = cft_cNEW
    if log:
        cft_c = np.log10(cft_c)
    msk = cft_c <= borne_inf
    cft_c[msk] = borne_inf
    msk = cft_c >= borne_sup
    cft_c[msk] = borne_sup

    fig,ax = plt.subplots(1,1,figsize=(15,7))
    colors = ax.pcolor(k,w,cft_c,cmap = 'jet')
##    colors = ax.pcolor(cft_c,cmap = 'jet')
    plt.grid()
    col = fig.colorbar(colors,orientation = 'horizontal')
    if axis != None:
        plt.axis(axis)
        print('AXIS = ', axis)
    if sumy:
        if sumybad:
            plt.title(r'$\sum_{\forall x}\left|\sum_{\forall y}W_{\omega,k_x}(x,y)\right|^2$ for $%s$ and $\Delta n=$%1.2f at T=%1.2f$\omega_p^{-1}$'%(field,Dn,t[Nt]))
        else:
            plt.title(r'$\sum_{\forall x,y}\left|W_{\omega,k_x}(x,y)\right|^2$ for $%s$ and $\Delta n=$%1.2fat T=%1.2f$\omega_p^{-1}$'%(field,Dn,t[Nt]))
    else:
        plt.title(r'$\sum_{\forall x}\left|W_{\omega,k_x}(x,y)\right|^2$ for $%s$ and $\Delta n=$%1.2f at T=%1.2f$\omega_p^{-1}$ for y = %i $\lambda_D$'%(field,Dn,t[Nt],ytranche))
    if log:
        col.ax.set_xlabel(r'$log_{10}$of energy of the mode  $(k_x,\omega)$',size = 13)
###        ax.set_yscale('log')
    else:
        col.ax.set_xlabel(r'Energy of the mode  $(k_x,\omega)$',size = 13)
#    grid = False
    if grid:
        plt.grid(b=True, which='major', color='w',alpha = 0.6,linestyle = '-',linewidth = 2);
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='w',alpha = 0.4,linestyle = '--',linewidth = 0.9);
    del cft_c
    return fig

def create_vid(cft,w,k,name,t,borne_inf = -10,borne_sup = -2,field = '',Dn = 0, log = False,fr = '',axis = None,logw = False,sumy = True,sumybad = False):
    if fr == '':
        fr = input('how many frames?')
    if field == '':
        field = input('What field ?')
    Nve = np.size(t)
    framestaked = np.arange(1,Nve,int(Nve/float(fr)))
    frr = np.size(framestaked)
    bar = Bar('Creating frames', max = frr+1,suffix='%(percent)d%%')
    for i,N in enumerate(framestaked):
        fig = dofig(cft,w,k,N,t,Dn=Dn,borne_inf = borne_inf,borne_sup = borne_sup,field = field,log = log,axis = axis,sumy = sumy,sumybad = sumybad);
        plt.savefig('frame %i.png'%(i))
        plt.close(fig)
        bar.next()
    with imageio.get_writer(name+'.mp4',fps = 5) as writer:
        avance_vec = np.arange(0,frr)
        bar = Bar('Writing mp4', max = frr,suffix='%(percent)d%% (%(elapsed)ds elapsed)')
        for i in range(0, len(framestaked)):
            bar.next()
            image = imageio.imread(f'frame {i}.png')
            writer.append_data(image)
            os.remove(f'frame {i}.png')

# def read_folder(path):
#     for root, directories, files in os.walk(path,topdown=True):
#         return files
# def rejoint(folder):
#     paths = read_folder(folder)
#     #### Rejoint if it's necesary
#     # Sorting by parts
#     key = lambda txt: np.int32(txt[txt.find('Part')+4:txt.find('of')])
#     try:
#         paths.sort(key = key)
#     except ValueError:
#         pass
#     # Joint
#     for s,path in enumerate(paths):
#         print('loading: '+path)
#         data = loadmat(folder+'/'+path)
#         if s == 0:
#             cft = data['Sum_cft_square']
#             t = data['t'][0]
#             w = data['w']
#             k = data['k']
#             try:
#                 omega0 = data['omega0']
#             except KeyError:
#                 omega0 = np.array([6,0])
#         else:
#             cft = np.concatenate((cft,data['Sum_cft_square']),axis =0)
#             t = np.concatenate((t,data['t'][0]),axis = 0)
#         del data
#     return cft,w,k,t    


# def dofigwtkt(cft,w,k,t,sumy = True,sumybad = False,field = '',Dn = 0,log = True,borne_inf = -3,borne_sup = 3,axiskt = None, axiswt = None,grid = True,logw = False):
#     cft[np.isnan(cft)] = 0
#     if field == '':
#         field = input('What field is it ?')
#     wvec = w[:,0]
#     kvec = k[0,:]
#     wt = np.sum(cft,axis = 2).T*np.abs(kvec[0]-kvec[1])
#     kt = np.sum(cft,axis = 1).T*np.abs(wvec[0]-wvec[1])
    
#     if log:
#         wt = np.log10(wt)
#         msk = wt <= borne_inf
#         wt[msk] = borne_inf
#         msk = wt >= borne_sup
#         wt[msk] = borne_sup
#         kt = np.log10(kt)
#         msk = kt <= borne_inf
#         kt[msk] = borne_inf
#         msk = kt >= borne_sup
#         kt[msk] = borne_sup
#     fig,ax = plt.subplots(2,1,figsize=(15,20))
#     if axiskt != None:
#         ax[1].axis(axiskt)
#     if axiswt != None:
#         ax[0].axis(axiswt)
#     ax[0].pcolor(t,wvec,wt,cmap = 'jet')
#     plt.grid()
#     colors = ax[1].pcolor(t,kvec,kt,cmap = 'jet')
#     col = fig.colorbar(colors,orientation = 'horizontal')
#     ax[0].set_ylabel('$\omega/\omega_p$',size = 15)
#     ax[1].set_ylabel('$k_x\lambda_D$',size = 15)
#     ax[1].set_xlabel('$t\omega_p$',size = 15)
#     if grid:
#         ax[0].grid(b=True, which='major', color='w',alpha = 0.6,linestyle = '-',linewidth = 2);
#         ax[0].minorticks_on()
#         ax[0].grid(b=True, which='minor', color='w',alpha = 0.4,linestyle = '--',linewidth = 0.9);
#         ax[1].grid(b=True, which='major', color='w',alpha = 0.6,linestyle = '-',linewidth = 2);
#         ax[1].minorticks_on()
#         ax[1].grid(b=True, which='minor', color='w',alpha = 0.4,linestyle = '--',linewidth = 0.9);
#     if sumy:
#         if sumybad:
#             ax[0].set_title(r'$\sum_{\omega}\;\sum_{\forall x}\left|\sum_{\forall y}W_{\omega,k_x}(x,y)\right|^2$ and $\sum_{k_x}\;\sum_{\forall x}\left|\sum_{\forall y}W_{\omega,k_x}(x,y)\right|^2$ for $%s$ and $\Delta n=$%1.2f'%(field,Dn))
#         else:
#             ax[0].set_title(r'$\sum_{\omega}\;\sum_{\forall x,y}\left|W_{\omega,k_x}(x,y)\right|^2$ and $\sum_{k_x}\;\sum_{\forall x,y}\left|W_{\omega,k_x}(x,y)\right|^2$ for $%s$ and $\Delta n=$%1.2f'%(field,Dn))
            
#     if log:
#         col.ax.set_xlabel(r'$log_{10}$of energy of all the modes all the modes with $k_x$ or $\omega$',size = 13)
#     else:
#         col.ax.set_xlabel(r'Energy of all the modes all the modes with $k_x$ or $\omega$',size = 13)
#     if logw:
#         ax[0].set_yscale('log')
#     return fig







# # def dofigwtkt(cft,w,k,t,sumy = True,field = '',Dn = 0,log = True,borne_inf = -3,borne_sup = 3,axiskt = None, axiswt = None,grid = True,logw = False,y_cut = 748):
# #     cft[np.isnan(cft)] = 0
# #     if field == '':
# #         field = input('What field is it ?')
# #     wvec = w[:,0]
# #     kvec = k[0,:]
# #     wt = np.mean(cft,axis = 2).T
# #     kt = np.mean(cft,axis = 1).T
    
# #     if log:
# #         wt = np.log10(wt)
# #         msk = wt <= borne_inf
# #         wt[msk] = borne_inf
# #         msk = wt >= borne_sup
# #         wt[msk] = borne_sup
# #         kt = np.log10(kt)
# #         msk = kt <= borne_inf
# #         kt[msk] = borne_inf
# #         msk = kt >= borne_sup
# #         kt[msk] = borne_sup
# #     fig,ax = plt.subplots(2,1,figsize=(15,20))
# #     if axiskt != None:
# #         ax[1].axis(axiskt)
# #     if axiswt != None:
# #         ax[0].axis(axiswt)
# #     ax[0].pcolor(t,wvec,wt,cmap = 'jet')
# #     plt.grid()
# #     colors = ax[1].pcolor(t,kvec,kt,cmap = 'jet')
# #     col = fig.colorbar(colors,orientation = 'horizontal')
# #     ax[0].set_ylabel('$\omega/\omega_p$',size = 15)
# #     ax[1].set_ylabel('$k_x\lambda_D$',size = 15)
# #     ax[1].set_xlabel('$t\omega_p$',size = 15)
# #     if grid:
# #         ax[0].grid(b=True, which='major', color='w',alpha = 0.6,linestyle = '-',linewidth = 2);
# #         ax[0].minorticks_on()
# #         ax[0].grid(b=True, which='minor', color='w',alpha = 0.4,linestyle = '--',linewidth = 0.9);
# #         ax[1].grid(b=True, which='major', color='w',alpha = 0.6,linestyle = '-',linewidth = 2);
# #         ax[1].minorticks_on()
# #         ax[1].grid(b=True, which='minor', color='w',alpha = 0.4,linestyle = '--',linewidth = 0.9);
# #     if sumy:
# #         ax[0].set_title(r'$\sum_{\omega}\;\sum_{\forall x,y}\left|W_{\omega,k_x}(x,y)\right|^2$ and $\sum_{k_x}\;\sum_{\forall x,y}\left|W_{\omega,k_x}(x,y)\right|^2$ for $%s$ and $\Delta n=$%1.2f'%(field,Dn)) 
# #     else:
# #         ax[0].set_title(r'$\sum_{\omega}\;\sum_{\forall x}\left|W_{\omega,k_x}(x,y)\right|^2$ and $\sum_{k_x}\;\sum_{\forall x,y}\left|W_{\omega,k_x}(x,y)\right|^2$ for $%s$, $\Delta n=$%1.2f, and y=%i$\lambda_D$'%(field,Dn,y_cut)) 
# #     if log:
# #         col.ax.set_xlabel(r'$log_{10}$of energy of all the modes all the modes with $k_x$ or $\omega$',size = 13)
# #     else:
# #         col.ax.set_xlabel(r'Energy of all the modes all the modes with $k_x$ or $\omega$',size = 13)
# #     if logw:
# #         ax[0].set_yscale('log')
# #     return fig

# def dofig(cft,w,k,Nt,t,axis = None,sumy = True,field = '',Dn = 0,log = True,borne_inf = -3,borne_sup = 3,grid = True,logw = False,y_cut = 748):
#     if field == '':
#         field = input('What field is it ?')
#     cft_c = cft[Nt,:,:].copy()
# #    if log:
#         # cft_c = np.log10(cft_c)
#         # msk = cft_c <= borne_inf
#         # cft_c[msk] = borne_inf
#         # msk = cft_c >= borne_sup
#         # cft_c[msk] = borne_sup
#     fig,ax = plt.subplots(1,1,figsize=(15,7))
#     colors = ax.pcolor(k,w,cft_c,cmap = 'jet')
#     plt.grid()
#     col = fig.colorbar(colors,orientation = 'horizontal')
#     if axis != None:
#         plt.axis(axis)
#     if sumy:
#         plt.title(r'$\sum_{\forall x,y}\left|W_{\omega,k_x}(x,y)\right|^2$ for $%s$ and $\Delta n=$%1.2fat T=%1.2f$\omega_p^{-1}$'%(field,Dn,t[Nt]))
#     else:
#         plt.title(r'$\sum_{\forall x}\left|W_{\omega,k_x}(x)\right|^2$ for $%s$ and $\Delta n=$%1.2f at T=%1.2f$\omega_p^{-1}$ for y = %i $\lambda_D$'%(field,Dn,t[Nt],y_cut))
#     if log:
#         col.ax.set_xlabel(r'$log_{10}$of energy of the mode  $(k_x,\omega)$',size = 13)
#     else:
#         col.ax.set_xlabel(r'Energy of the mode  $(k_x,\omega)$',size = 13)
#     if logw:
#         ax.set_yscale('log')
#     if grid:
#         plt.grid(b=True, which='major', color='w',alpha = 0.6,linestyle = '-',linewidth = 2);
#         plt.minorticks_on()
#         plt.grid(b=True, which='minor', color='w',alpha = 0.4,linestyle = '--',linewidth = 0.9);
#     del cft_c
#     return fig

# def create_vid(cft,w,k,name,t,borne_inf = -10,borne_sup = -2,field = '',Dn = 0, log = False,fr = '',axis = None,logw = False,sumy = True):
#     if fr == '':
#         fr = input('how many frames?')
#     if field == '':
#         field = input('What field ?')
#     Nve = np.size(t)
#     framestaked = np.arange(1,Nve,int(Nve/float(fr)))
#     frr = np.size(framestaked)
#     i = 0
#     bar = Bar('Writing mp4', max = frr,suffix='%(percent)d%% (%(elapsed)ds elapsed)')
#     print('Doing frames of the video\n')
#     for N in framestaked:
#         fig = dofig(cft,w,k,N,t,Dn=Dn,borne_inf = borne_inf,borne_sup = borne_sup,field = field,log = log,axis = axis,logw = logw,sumy = sumy);
#         plt.savefig('frame %i.png'%(i))
#         plt.close(fig)
#         i += 1
#         bar.next()
#     print('Constructing mp4\n')
#     with imageio.get_writer(name+'.mp4',fps = 5) as writer:
#         bar = Bar('Writing mp4', max = frr,suffix='%(percent)d%% (%(elapsed)ds elapsed)')
#         for i in range(0, len(framestaked)):
#             bar.next()
#             image = imageio.imread(f'frame {i}.png')
#             writer.append_data(image)
#             os.remove(f'frame {i}.png')

def listFile(folder_to_write, file_names):
    # donne la liste des fichier de meme nom pour ne pas les ecraser
    listFichier = glob.glob(folder_to_write+os.sep+'*'+file_names+'*')
    if (len(listFichier) > 1):
        incre=[]
        for ii in range(len(listFichier)):
            try:
                incre.append(int(listFichier[ii][listFichier[ii].index('__')+2:listFichier[ii].index('.png')]))
            except:
                incre=[1]
                pass
    else:
        incre=[1]
    increment = np.max(incre)
    return str(increment+1)

def read_folder(path):
    import glob
    files = glob.glob(path+os.sep+'*')
#    for root, directories, files in os.walk(path,topdown=True):
    return files

def rejoint(folder):
    paths = read_folder(folder)
    #### Rejoint if it's necesary
    # Sorting by parts
    paths.sort()
    # Joint
    for s,path in enumerate(paths):
#        print('chemin !!!! =', path)
        print('s = ', s)
#        data = loadmat(folder+'/'+path)
        data = loadmat(path)
        if s == 0:
            cft = data['Sum_cft_square']
            t = data['t'][0]
            w = data['w']
            k = data['k']
            coefficient = data['Normalization_coeficient']
            try:
                omega0 = data['omega0']
            except KeyError:
                omega0 = np.array([6,0])
        else:
            cft = np.concatenate((cft,data['Sum_cft_square']),axis =0)
            t = np.concatenate((t,data['t'][0]),axis = 0)
        del data
    return cft,w,k,t,coefficient

def waveletVisualization(Folder_to_read_dotmat, folder_to_write):
    ### GETTING DATA
    Sum_xy_cft_square,w,k,t,coefficient = rejoint(Folder_to_read_dotmat)
    
    ### GETTING PARAMETERS FROM NAME
    
    for roots,directorys,files in os.walk(Folder_to_read_dotmat):
            file = files[0]
            parts = len(files)
    field = file[file.find('FICIENTS_')+9:file.find('_with_Dn')]
    Dn = np.float32(file[file.find('Dn=')+3:file.find('_data_for')])
    wmin = np.float32(file[file.find('wmin=')+5:file.find('_wmax')])
    wmax = np.float32(file[file.find('wmax=')+5:file.find('_kmin')])
    kmin = np.float32(file[file.find('kmin=')+5:file.find('_kmax')])
    kmax = np.float32(file[file.find('kmax=')+5:file.find('_Dw')])
    Dw   = np.float32(file[file.find('_Dw=')+4:file.find('_Dk')])
    Dk   = np.float32(file[file.find('_Dk=')+4:file.find('_T=')])
    T_min = t[0]
    T_max = t[-1]
    
    log = True
    logw = False
    sumy = True
    sumybad=True

    borne_inf_colorbar = inputData.dict_Wavelet['MinColorBar']
    borne_sup_colorbar = inputData.dict_Wavelet['MaxColorBar']
    view_kmin = inputData.dict_Wavelet['k_min']
    view_kmax = inputData.dict_Wavelet['k_max']
    view_Wmin = inputData.dict_Wavelet['omega_min']
    view_Wmax = inputData.dict_Wavelet['omega_max']
    view_Tmin = inputData.dict_Wavelet['T_init']
    view_Tmax = inputData.dict_Wavelet['T_fin']
    
    number_of_frames = 100
    
#    axis = [kmin,kmax,wmin,wmax]

    axis = [view_kmin,view_kmax,view_Wmin,view_Wmax]

# on teste l'existence du repertoire folder_to_write s'il n'existe pas on le cree et s'il existe on ne fait rien.
    if (not os.path.isdir(folder_to_write)):
        os.mkdir(folder_to_write)
###    print('borne_sup_colorbar = ', borne_sup_colorbar)
    fig = dofig(Sum_xy_cft_square,w,k,np.int32(len(t)/2),t,axis = axis,sumy = sumy,field = field,Dn = Dn,log = log,borne_inf = borne_inf_colorbar,borne_sup = borne_sup_colorbar)
#    plt.show()
    name_figDISPERSION = 'w(k)for_%s_with_Dn=%1.2f_analysis_for_wmin=%1.3f_wmax=%1.3f_kmin=%1.3f_kmax=%1.3f_Dw=%1.3f_Dk=%1.3f_from_T=%i_to_T=%i'%(field,Dn,view_Wmin,view_Wmax,view_kmin,view_kmax,Dw,Dk,view_Tmin,view_Tmax)
# si la figure existe deja, on incremente sa valeur pour ne pas l'ecraser
    increment = listFile(folder_to_write, name_figDISPERSION)

    fig.savefig(folder_to_write+os.sep+name_figDISPERSION+'__'+increment+'.png')
    yn = input('Do you like the figure?(y/n)\n')
    if yn.capitalize() == 'Y':
        type = input('1 : Do video\n2 : Do video and fig k(t) and w(t)\n3 : Do only k(t) and w(t)\n4 : Do nothing\n')
    
        name_video = 'VIDEO_OF__%s_with_Dn=%1.2f_analysis_for_wmin=%1.3f_wmax=%1.3f_kmin=%1.3f_kmax=%1.3f_Dw=%1.3f_Dk=%1.3f_from_T=%i_to_T=%i'%(field,Dn,view_Wmin,view_Wmax,view_kmin,view_kmax,Dw,Dk,view_Tmin,view_Tmax)
        name_fig = 'w(t)_and_k(t)_for_%s_with_Dn=%1.2f_analysis_for_wmin=%1.3f_wmax=%1.3f_kmin=%1.3f_kmax=%1.3f_Dw=%1.3f_Dk=%1.3f_from_T=%i_to_T=%i'%(field,Dn,view_Wmin,view_Wmax,view_kmin,view_kmax,Dw,Dk,view_Tmin,view_Tmax)
    
        if type == '1':
            create_vid(Sum_xy_cft_square,w,k,folder_to_write+'/'+name_video,t,axis = axis,sumy = sumy,
                       field = field,Dn = Dn,log = log,borne_inf = borne_inf_colorbar,
                       borne_sup = borne_sup_colorbar,logw = log,fr = number_of_frames)
        elif type == '2':
            create_vid(Sum_xy_cft_square,w,k,folder_to_write+'/'+name_video,t,axis = axis,
                       sumy = sumy,field = field,Dn = Dn,log = log,borne_inf = borne_inf_colorbar,
                       borne_sup = borne_sup_colorbar,logw = log,fr = number_of_frames)
            
            fig = dofigwtkt(Sum_xy_cft_square,w,k,t,sumy = sumy,sumybad = sumybad,field = field,Dn = Dn,
                  logw = logw,log = log,borne_inf = borne_inf_colorbar,borne_sup = borne_sup_colorbar,
                  axiskt = [T_min,T_max,-0.3,0.3],axiswt = [T_min,T_max,0,4.5])
            
            # fig = dofigwtkt(Sum_xy_cft_square,w,k,t,sumy = sumy,field = field,Dn = Dn,logw = logw,
            #                 log = log,borne_inf = borne_inf_colorbar,borne_sup = borne_sup_colorbar,
            #                 axiskt = [t[0],t[-1],axis[0],axis[1]],axiswt = [t[0],t[-1],axis[2],axis[3]])
            fig.savefig(folder_to_write+'/'+name_fig+'.png')
        elif type == '3':
            fig = dofigwtkt(Sum_xy_cft_square,w,k,t,sumy = sumy,sumybad = sumybad,field = field,Dn = Dn,
                  logw = logw,log = log,borne_inf = borne_inf_colorbar,borne_sup = borne_sup_colorbar,
                  axiskt = [view_Tmin,view_Tmax,view_kmin, view_kmax],axiswt = [view_Tmin,view_Tmax,view_Wmin,view_Wmax])
            
            # fig = dofigwtkt(Sum_xy_cft_square,w,k,t,sumy = sumy,field = field,Dn = Dn,logw = logw,
            #                 log = log,borne_inf = borne_inf_colorbar,borne_sup = borne_sup_colorbar,
            #                 axiskt = [t[0],t[-1],axis[0],axis[1]],axiswt = [t[0],t[-1],axis[2],axis[3]])
# si la figure existe deja, on incremente sa valeur pour ne pas l'ecraser
            increment = listFile(folder_to_write, name_fig)
            fig.savefig(folder_to_write+os.sep+name_fig+'__'+increment+'.png')
            print('Figure done !!')
        else:
            print('No figure was created')
    else:
        print('No figure was created')
