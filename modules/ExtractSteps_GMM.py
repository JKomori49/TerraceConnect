import rasterio
from rasterio.transform import from_origin
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.utils import resample
import matplotlib.pyplot as plt
import time

def ReadParamFile(file_path):
    # Dictionary to store the variables
    variables = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and skip empty lines or lines starting with '#'
            line = line.strip()
            if line and not line.startswith('#'):
                # Split the line at the first occurrence of whitespace
                key, value = line.split(maxsplit=1)
                value = float(value)
                # Store in dictionary
                variables[key] = value
    
    # Dynamically create variables from the dictionary
    for key, value in variables.items():
        # Assign variables in the local scope of the calling function
        exec(f"{key} = {value}", None, globals())
    
    # Optionally, return the dictionary of variables
    return variables

def find_segments(P, r, dz):
    nrows = len(P)
    dx = int(r/(P[1,0]-P[0,0]))
    segments = []
    in_segment = False
    
    for x in range(nrows - dx-1):  # Ensure x + r is within bounds
        if np.all(P[x+dx,1] < P[x,1] + dz):  # Condition check
            if not in_segment:
                segment_start = x
                in_segment = True
        else:
            if in_segment:
                segments.append((P[x,0], P[x,1]))
                in_segment = False

    # Check if the last checked segment was still open
    if in_segment:
        segments.append((P[x,0], P[x,1]))
    
    return segments

def find_segments_write(P, r, dz):
    nrows = len(P)
    dx = int(r/(P[1,0]-P[0,0]))
    segments = []
    segments_out = []
    in_segment = False
    
    for x in range(nrows - dx-1):  # Ensure x + r is within bounds
        if np.all(P[x+dx,1] < P[x,1] + dz):  # Condition check
            segments_out.append((P[x,0], 0.))
            if not in_segment:
                segment_start = x
                in_segment = True
        else:
            segments_out.append((P[x,0], P[x,1]))
            if in_segment:
                segments.append((P[x,0], P[x,1]))
                in_segment = False

    # Check if the last checked segment was still open
    if in_segment:
        segments.append((P[x,0], P[x,1]))
    
    return segments,segments_out

def cliffs(src,r,dz,Xmin,Xmax,Hmin,Hmax,dw,margin,fill):
    array = src.read(1)
    transform = src.transform
    
    Nt = array.shape[1]
    dx = transform.a
    dy = abs(transform.e)
    xaxis = dx * np.arange(Nt)
    
    Iw = round(dw/dy)
    Imin = round(Xmin/dy)
    Imax = round(Xmax/dy)
    
    extracted = np.array([0,0,0])
    for vert in range(Imin,Imax,Iw):
        transect = np.column_stack((xaxis,array[vert,:]))
        
        if fill == 1:
            for i in range(1,len(transect)):
                if transect[i,1] < transect[i-1,1]:
                    transect[i,1] = transect[i-1,1]

        Hlim = len(transect)
        for i in range(1,len(transect)):
            if transect[i,1] > Hmax+margin:
                Hlim = i
                break
        transect = transect[:Hlim,:]
        
        ts_f = transect[(transect[:,1] > Hmin - margin) & (transect[:,1] < Hmax + margin)]
        
        if(len(ts_f)>1):
            segments = find_segments(ts_f,r,dz)
            if len(segments)>0:
                segments = np.array(segments)
                prof_position = np.full((segments.shape[0], 1), vert*dy)
                extracted = np.vstack(( extracted, np.hstack((prof_position, segments)) ))

    extracted = extracted[1:,:]
    filtered_array = extracted[(extracted[:,2] > Hmin) & (extracted[:,2] < Hmax)]
    return filtered_array

def cliffs_single(src,r,dz,Hmin,Hmax,dw,position,fill):
    array = src.read(1)
    transform = src.transform
    
    margin = 2

    Nt = array.shape[1]
    dx = transform.a
    dy = abs(transform.e)
    xaxis = dx * np.arange(Nt)
    
    extracted = np.array([0,0,0])
    vert = round(position/dy)
    if (vert<0) or (vert>array.shape[0]):
        print("position out of range")

    transect = np.column_stack((xaxis,array[vert,:]))
    transect_copy = np.copy(transect)
    #Fill pitholes
    if fill==1:
        for i in range(1,len(transect)):
            if transect[i,1] < transect[i-1,1]:
                transect[i,1] = transect[i-1,1]
    Hlim = len(transect)
    for i in range(1,len(transect)):
        if transect[i,1] > Hmax+margin:
            Hlim = i
            break
    transect = transect[:Hlim,:]

    ts_f = transect[(transect[:,1] > Hmin-margin) & (transect[:,1] < Hmax+margin)]
    ts_c = transect_copy[(transect_copy[:,1] > Hmin) & (transect_copy[:,1] < Hmax)]
        
    #segments = find_segments(ts_f,r,dz)
    segments,segments_out = find_segments_write(ts_f,r,dz)
    segments_out = np.array(segments_out)

    if len(segments)>0:
        segments = np.array(segments)
        prof_position = np.full((segments.shape[0], 1), vert*dy)
        extracted = np.vstack(( extracted, np.hstack((prof_position, segments)) ))

    extracted = extracted[1:,:]
    filtered_array = extracted[(extracted[:,2] > Hmin) & (extracted[:,2] < Hmax)]
    
    #plt.plot(ts_f[:,0],ts_f[:,1])
    plt.plot(ts_c[:,0],ts_c[:,1])
    #plt.plot(segments_out[:,0],segments_out[:,1])
    plt.scatter(filtered_array[:,1],filtered_array[:,2],c='green',marker ='o', s = 20.0)
    plt.show()
    
def GMMshowresult(data,K):
    gmm = GaussianMixture(n_components=K, covariance_type='full', random_state=42)
    gmm.fit(data.reshape(-1, 1))
    
    labels = gmm.predict(data.reshape(-1, 1))
    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    weights = gmm.weights_
    
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=60, density=True, alpha=0.6, color='g')

    x = np.linspace(min(data), max(data), 1000)
    for mean, cov, weight in zip(means, covariances, weights):
        plt.plot(x, weight * (1 / np.sqrt(2 * np.pi * cov)) * np.exp(-0.5 * (x - mean)**2 / cov), linewidth=2)

    plt.show()
    
def GMMoutmean(data,K):
    gmm = GaussianMixture(n_components=K, covariance_type='full', random_state=42)
    gmm.fit(data.reshape(-1, 1))
    
    labels = gmm.predict(data.reshape(-1, 1))
    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    weights = gmm.weights_
    
    return means

def SingleSection(extracted,W,d,position):
    Rmin = min(extracted[:,0])
    Rmax = max(extracted[:,0])
    L = Rmax-Rmin
    
    NI = 100

    Result = []
    PRCS = 0
    STEP = 10

    x0 = position-W/2
    x1 = x0+W
    AnalysisWindow = extracted[(x0 <= (extracted[:,0])) & ((extracted[:,0]) < x1)]
    Np = len(AnalysisWindow)
        
    n_components_range = range(2, 10)
        
    y = AnalysisWindow[:,2]
    aic = []
    
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(y.reshape(-1, 1))
        aic.append(gmm.aic(y.reshape(-1, 1)))
    
    optimal_n = n_components_range[np.argmin(aic)]
    print(optimal_n)
    GMMshowresult(y,optimal_n)
    means = GMMoutmean(y,optimal_n)
    print(means)
    
    #plt.plot(n_components_range, aic, marker='o')
    #plt.legend()
    #plt.show()
    
def SingleSection_resample(extracted,W,d,position):
    Rmin = min(extracted[:,0])
    Rmax = max(extracted[:,0])
    L = Rmax-Rmin
    
    Result = []
    PRCS = 0
    STEP = 10

    x0 = position-W/2
    x1 = x0+W
    AnalysisWindow = extracted[(x0 <= (extracted[:,0])) & ((extracted[:,0]) < x1)]
    Np = len(AnalysisWindow)
        
    n_components_range = range(2, 10)
        
    y = AnalysisWindow[:,2]
    aic = []
    y_resample = resample(y)
    
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(y_resample.reshape(-1, 1))
        aic.append(gmm.aic(y_resample.reshape(-1, 1)))
    
    GMMshowresult(y_resample,7)
    optimal_n = n_components_range[np.argmin(aic)]
    print(optimal_n)
    means = GMMoutmean(y_resample,optimal_n)
    print(means)
    
    #plt.plot(n_components_range, aic, marker='o')
    #plt.legend()
    #plt.show()
    
def SingleSection_Bootstrap(extracted,W,d,position,n_bootstrap,Kmin,Kmax):
    Rmin = min(extracted[:,0])
    Rmax = max(extracted[:,0])
    L = Rmax-Rmin
    
    NI = 100
    
    Result = []
    PRCS = 0
    STEP = 10

    x0 = position-W/2
    x1 = x0+W
    AnalysisWindow = extracted[(x0 <= (extracted[:,0])) & ((extracted[:,0]) < x1)]
    Np = len(AnalysisWindow)
        
    #plt.scatter(AnalysisWindow[:,0],AnalysisWindow[:,2],marker ='o', s = 1.0)
    
    n_components_range = range(Kmin, Kmax)
        
    y = AnalysisWindow[:,2]
    #plt.hist(y, bins=50, range=(0,500), density=True, alpha=0.6, color='g')
    
    for bs in range(n_bootstrap):
        y_resample = resample(y)
        
        aic = []
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
            gmm.fit(y_resample.reshape(-1, 1))
            aic.append(gmm.aic(y_resample.reshape(-1, 1)))
    
        optimal_n = n_components_range[np.argmin(aic)]
        means = GMMoutmean(y_resample,optimal_n)
        
        for i in range(len(means)):
            Result.append(means[i])
            
        if (bs*100)/n_bootstrap > PRCS:
            PRCS += STEP
            print(f"{PRCS}% complete")
    Result = np.array(Result)
    plt.hist(Result, bins=100, range=(0,50), density=False, alpha=0.6, color='g')
    counts, bin_edges = np.histogram(Result, bins=50, range=(0,500))
    print(counts)

#    plt.legend()
    plt.show()
    
def MultiSections_Bootstrap(extracted,W,d,n_bootstrap,Kmin,Kmax,Hmin,Hmax,Nbin):
    Rmin = min(extracted[:,0])
    Rmax = max(extracted[:,0])
    
    L = Rmax-Rmin
    N = int((L-W)//d)
    
    n_components_range = range(Kmin,Kmax)
        
    PRCS = 0
    STEP = 10
    Result = []

    NoresultRange = int((Rmin - W/2)//d)
    ResultZero = np.zeros(Nbin)
    for i in range(NoresultRange):
        Result.append(ResultZero)

    for i in range(N):
        x0 = Rmin + d*(i-1)
        x1 = x0+W
        AnalysisWindow = extracted[(x0 <= (extracted[:,0])) & ((extracted[:,0]) < x1)]
        Np = len(AnalysisWindow)
        
        y = AnalysisWindow[:,2]
        hist = []
        for bs in range(n_bootstrap):
            y_resample = resample(y)
            aic = []
            for n_components in n_components_range:
                gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
                gmm.fit(y_resample.reshape(-1, 1))
                aic.append(gmm.aic(y_resample.reshape(-1, 1)))
    
            optimal_n = n_components_range[np.argmin(aic)]
        
            means = GMMoutmean(y_resample,optimal_n)
        
            for j in range(len(means)):
                hist.append(means[j])
                
        hist = np.array(hist)
        counts, bin_edges = np.histogram(hist, bins=Nbin, range=(Hmin,Hmax))
        Result.append(counts)

        if (i*100)/N > PRCS:
            PRCS += STEP
            print(f"{PRCS}% complete")
    
    Result = np.array(Result)
    np.savetxt('result.txt', Result, fmt='%d')
    
    #plt.imshow(Result.T, cmap='plasma')
    #plt.show()
    return(Result)

def MultiSections(extracted,W,d,Kmin,Kmax):
    Rmin = min(extracted[:,0])
    Rmax = max(extracted[:,0])
    L = Rmax-Rmin
    N = int((L-W)//d)
    
    n_components_range = range(Kmin,Kmax)
    
    PRCS = 0
    STEP = 10
    Result = []

    for i in range(N):
        x0 = Rmin + d*(i-1)
        x1 = x0+W
        AnalysisWindow = extracted[(x0 <= (extracted[:,0])) & ((extracted[:,0]) < x1)]
        Np = len(AnalysisWindow)
        
        y = AnalysisWindow[:,2]
        aic = []
    
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
            gmm.fit(y.reshape(-1, 1))
            aic.append(gmm.aic(y.reshape(-1, 1)))
    
        optimal_n = n_components_range[np.argmin(aic)]
        
        means = GMMoutmean(y,optimal_n)
        
        for j in range(len(means)):
            Result.append((x0+W/2,means[j]))
        if (i*100)/N > PRCS:
            PRCS += STEP
            print(f"{PRCS}% complete")
    Result = np.array(Result)

    return(Result)
        
def main(DATAMAP, DATAPARAM, outfile):
    time01 = time.time()
    
    variables = ReadParamFile(DATAPARAM)
    
    r = variables['r']
    dz = variables['dz']
    Hmin = variables['Hmin']
    Hmax = variables['Hmax']
    dw = variables['dw']
    
    position = 900
    Xmin = 0
    Xmax = 2300
    

    with rasterio.open(DATAMAP) as src:
        #extracted = cliffs(src,r,dz,Xmin,Xmax,Hmin,Hmax,dw,margin=5,fill=1)
        cliffs_single(src,r,dz,Hmin,Hmax,dw,position,fill=1)
    #print(len(extracted))


    time02 = time.time()
    
    print(f"Cliff extraction: {(time02 - time01)/60:.5f} mins")
    #plt.scatter(extracted[:,0],extracted[:,2],marker ='o', s = 1.0)
    d = variables['dx']
    W = variables['Wr']

    #SingleSection(extracted,W,d,position)
    #SingleSection_resample(extracted,W,d,position)
    #SingleSection_Bootstrap(extracted,W,d,position,400,2,10)
    
    #Result = MultiSections_Bootstrap(extracted,W,d,100,2,11,Hmin,Hmax,400)
    
    #Result = MultiSections(extracted,W,d,2,7)
    
    #plt.scatter(Result[:,0],Result[:,1],marker ='o', s = 1.0)
    
    #plt.show()
    
    time03 = time.time()
    print(f"Classification: {(time03 - time02)/60:.5f} mins")
