import rasterio
from rasterio.transform import from_origin
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.utils import resample
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.cm as cm
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

def cliffs(src,r,dz,Xmin,Xmax,Hmin,Hmax,dw,fill):
    array = src.read(1)
    transform = src.transform
    
    margin = (Hmax-Hmin)/10

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
    
    margin = (Hmax-Hmin)/10

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
        
    segments,segments_out = find_segments_write(ts_f,r,dz)
    segments_out = np.array(segments_out)

    if len(segments)>0:
        segments = np.array(segments)
        prof_position = np.full((segments.shape[0], 1), vert*dy)
        extracted = np.vstack(( extracted, np.hstack((prof_position, segments)) ))

    extracted = extracted[1:,:]
    filtered_array = extracted[(extracted[:,2] > Hmin) & (extracted[:,2] < Hmax)]
    
    plt.plot(ts_c[:,0],ts_c[:,1])
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

def GMMout(data,K):
    gmm = GaussianMixture(n_components=K, covariance_type='full', random_state=42)
    gmm.fit(data.reshape(-1, 1))
    
    labels = gmm.predict(data.reshape(-1, 1))
    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    weights = gmm.weights_
    
    return means,covariances,weights

def SingleSection(extracted,W,d,position,Kmin,Kmax):
    x0 = position-W/2
    x1 = x0+W
    AnalysisWindow = extracted[(x0 <= (extracted[:,0])) & ((extracted[:,0]) < x1)]
    Np = len(AnalysisWindow)
        
    n_components_range = range(Kmin, Kmax)
    y = AnalysisWindow[:,2]
    
    # Create a figure and two subplots (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.scatter(AnalysisWindow[:,0],AnalysisWindow[:,2],c='blue',marker ='o', s = 1.0)
    ax2.hist(y, bins=60, orientation='horizontal', density=True, alpha=0.6, color='g')
         
    aic = []
    
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(y.reshape(-1, 1))
        aic.append(gmm.aic(y.reshape(-1, 1)))
    
    optimal_n = n_components_range[np.argmin(aic)]
    
    means, covariances, weights = GMMout(y,optimal_n)
    
    x = np.linspace(min(y), max(y), 1000)
    for mean, cov, weight in zip(means, covariances, weights):
        ax2.plot(weight * (1 / np.sqrt(2 * np.pi * cov)) * np.exp(-0.5 * (x - mean)**2 / cov), x, linewidth=2)
    
    # Adjust layout for better display
    plt.tight_layout()

    # Show both plots
    plt.show()
    
def SingleSection_Bootstrap(extracted,W,d,position,n_bootstrap,Kmin,Kmax):
    Result = []
    PRCS = 0
    STEP = 10
    
    x0 = position-W/2
    x1 = x0+W
    AnalysisWindow = extracted[(x0 <= (extracted[:,0])) & ((extracted[:,0]) < x1)]
    Np = len(AnalysisWindow)
    
    n_components_range = range(Kmin, Kmax)
    
    # Create a figure and two subplots (side by side)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
    ax1.scatter(AnalysisWindow[:,0],AnalysisWindow[:,2],c='blue',marker ='o', s = 1.0)
    ax1.set_xlabel('Distance along profile (x) [m]')
    ax1.set_ylabel('Elevation [m]')
    
    y = AnalysisWindow[:,2]
    ax2.hist(y, bins=60, orientation='horizontal', density=True, alpha=0.6, color='g')
    ax2.set_xlabel('Probability density')
    
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
    scale_factor = n_bootstrap/100
    ax3.hist(Result, bins=100,weights=np.ones_like(Result) / scale_factor, orientation='horizontal',density=False, alpha=0.6, color='b')
    ax3.set_xlabel('Detection probability [%]')
    
    # Adjust layout for better display
    plt.tight_layout()

    # Show all plots
    plt.show()
    
def MultiSections_Bootstrap(extracted,W,d,n_bootstrap,Kmin,Kmax,Hmin,Hmax,zint):
    Rmin = min(extracted[:,0])
    Rmax = max(extracted[:,0])
    
    L = Rmax-Rmin
    N = int((L-W)//d)
    
    n_components_range = range(Kmin,Kmax)
        
    PRCS = 0
    STEP = 10
    Result = []

    NoresultRange = int((Rmin - W/2)//d)

    Nbin = int((Hmax-Hmin)/zint)
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
    Resultnorm = Result/n_bootstrap*100

    ResultShow = np.flipud(Resultnorm.T)
    extent = [W/2, Rmax-W/2, Hmin, Hmax]

    # Define the two colormaps (grayscale and viridis)
    Th = int(256*0.3)
    grayscale = LinearSegmentedColormap.from_list('grayscale', ['white', 'black'], N=256)
    viridis = cm.get_cmap('viridis', 256-Th)

    # Combine the two colormaps into one
    colors = np.vstack((
        grayscale(np.linspace(0, 1, 256))[:Th],  # Grayscale from 0 to 0.3
        viridis(np.linspace(1, 0, 256-Th))     # Viridis from 0.3 to 1.0
    ))

    # Create the combined colormap
    combined_cmap = LinearSegmentedColormap.from_list('combined_cmap', colors)

    plt.figure(figsize=(8,6))
    plt.imshow(ResultShow, cmap=combined_cmap, aspect='auto',extent=extent)
    # Add a color bar to show the intensity scale
    plt.colorbar()

    # Add labels and title
    plt.title('Detection probability [%]')
    plt.xlabel('Distance along coast (x) [m]')
    plt.ylabel('Elevation [m]')

    # Show the plot
    plt.tight_layout()
    plt.show()

    return Resultnorm
    

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
        
def main(DATAMAP,DATAPARAM):
    time01 = time.time()
    
    MAPFILE=f"Warped//Warped_{DATAMAP}"
    PARAMFILE=f"config//{DATAPARAM}"

    variables = ReadParamFile(PARAMFILE)
    
    r = variables['r']
    dz = variables['dz']
    Hmin = variables['Hmin']
    Hmax = variables['Hmax']
    dw = variables['dw']
    
    mode = int(variables['mode'])
    fill = int(variables['Fill'])
    position = variables['position']
    
    #Get X range
    with rasterio.open(MAPFILE) as src:
        transform = src.transform
        height = src.height
        max_y = transform[5]  # transform[5] is the y-coordinate of the top-left corner
        min_y = transform[5] + (transform[4] * height)  # transform[4] is usually a negative scale (vertical interval)
    Xmin = max_y
    Xmax = -min_y
    
    if mode==1:
        with rasterio.open(MAPFILE) as src:
            cliffs_single(src,r,dz,Hmin,Hmax,dw,position,fill)
        time02 = time.time()

    else:
        with rasterio.open(MAPFILE) as src:
            extracted = cliffs(src,r,dz,Xmin,Xmax,Hmin,Hmax,dw,fill)
        print(f"{len(extracted)} points extracted")
        time02 = time.time()
        print(f"Cliff extraction: {(time02 - time01)/60:.5f} mins")
        d = variables['dx']
        W = variables['Wr']
        n_bootstrap = int(variables['n_bootstrap'])
        Kmin = int(variables['Kmin'])
        Kmax = int(variables['Kmax'])
        zint = variables['zint']

        if mode==2:
            SingleSection(extracted,W,d,position,Kmin,Kmax)
        elif mode==3:
            SingleSection_Bootstrap(extracted,W,d,position,n_bootstrap,Kmin,Kmax)
        elif mode==4:
            temp = DATAMAP.replace(".tif", ".dat")
            OUTFILE=f"OUT//{temp}"
            Result = MultiSections_Bootstrap(extracted,W,d,n_bootstrap,Kmin,Kmax,Hmin,Hmax,zint)
            np.savetxt(OUTFILE, Result, fmt='%d')
        time03 = time.time()
        print(f"Classification: {(time03 - time02)/60:.5f} mins")
