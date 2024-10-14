import rasterio
from rasterio.transform import from_origin
from affine import Affine
import cupy as cp
import numpy as np
import math
from scipy.interpolate import splev, splprep
import time

def ReadProfileFile(DATAPROFILE):
    # Open the profile data and interpolating
    control_points = []
    with open(DATAPROFILE, 'r') as file:
        lines = file.readlines()
        # Read the range from the first line 
        range_line = lines[0].strip().split()
        R = [float(range_line[1]), float(range_line[2])]

        # Read the control points
        for line in lines[2:]:  # Start reading from the third line
            if line.strip() and not line.startswith('#'):
                # Split the line into components and convert them to integers
                point = list(map(float, line.strip().split()))
                control_points.append(point)
    P = np.array(control_points)
    return R,P

def Curvelength(P):
    #P: Polyline array(Np,2)
    #D: Output Position of each point(Np)
    Np = P.shape[0]
    D = np.zeros(Np)
    W = 0
    for i in range(1,Np):
        L = math.sqrt((P[i,0]-P[i-1,0])*(P[i,0]-P[i-1,0])+(P[i,1]-P[i-1,1])*(P[i,1]-P[i-1,1]))
        W += L
        D[i] = W
    return D

def Resample(P,d):
    #P: input polyline array(Np,2)
    #d: input resample distance
    #Q: output resampled polyline array(Nq,2)
    Np = P.shape[0]
    D = Curvelength(P)
    Nq = int(D[Np-1]/d)+2
    Q = np.zeros((Nq,2))
    Q[0,:] = P[0,:]
    Vertex_n = 0
    for i in range(1,Nq-1):
        dsum = d*i
        while dsum > D[Vertex_n]:
            Vertex_n += 1
            if Vertex_n == Np-1:
                break
        x1 = P[Vertex_n-1,0]
        y1 = P[Vertex_n-1,1]
        x2 = P[Vertex_n,0]
        y2 = P[Vertex_n,1]
        a = (dsum-D[Vertex_n-1])/(D[Vertex_n]-D[Vertex_n-1])
        Q[i,0] = x1*(1-a) + x2*a
        Q[i,1] = y1*(1-a) + y2*a
    Q[Nq-1,:] = P[Np-1,:]
    return Q

def bspline(P,d):
    x = P[:,0]
    y = P[:,1]
    # Create a B-spline representation
    
    # Spline parameters:
    # s = 0: spline will pass through all the points
    tck, u = splprep([x, y], s=1)
    N_fine = 1000
    u_fine = np.linspace(0, 1, N_fine)
    x_fine, y_fine = splev(u_fine, tck)
    
    # Resample with a Constant Distance
    L = Curvelength(np.column_stack([x_fine, y_fine]))
    Np = int(L[-1] // d)
    dp = L[-1]/Np
    
    # Define the number of points or the interval of arc length
    u_new = np.linspace(0, 1, Np)

    # Evaluate the spline at the new u values
    x_new, y_new = splev(u_new, tck)
    Q = np.column_stack([x_new, y_new])

    return Q, dp

def normal(P):
    # Compute the differences (dx, dy)
    tangents = np.diff(P, axis=0)
    # Calculate the averages of consecutive elements
    averages = (tangents[:-1, :] + tangents[1:, :]) / 2
    # Initialize Q with one extra row compared to P
    Q = np.zeros((tangents.shape[0] + 1, tangents.shape[1]))
    # Set the first and last rows of Q
    Q[0, :] = tangents[0, :]
    Q[-1, :] = tangents[-1, :]
    # Fill in the intermediate values
    Q[1:-1, :] = averages
    
    # Calculate the normals by rotating 90 degrees
    normals = np.array([-Q[:, 1], Q[:, 0]]).T
    
    # Normalize the normals for uniform length
    norm_length = np.sqrt(normals[:, 0]**2 + normals[:, 1]**2)
    normals[:, 0] /= norm_length
    normals[:, 1] /= norm_length

    return normals

def GetZ_batch_gpu(src, coords):
    """
    Fetch values for a batch of coordinates from a raster source using GPU acceleration.

    Parameters:
    - src: rasterio open dataset
    - coords: List of (x, y) tuples

    Returns:
    - List of values corresponding to the input coordinates
    """
    # Convert geographic coordinates to pixel coordinates
    pixels = [src.index(x, y) for x, y in coords]
    
    # Initialize an array to hold the results
    values = cp.full(len(coords), -999, dtype=cp.float32)  # Use appropriate dtype
    
    # Determine which pixel coordinates are within bounds
    valid_indices = [(i, px, py) for i, (px, py) in enumerate(pixels)
                     if 0 <= px < src.height and 0 <= py < src.width]
    
    print("start conversion")
    
    # Read the data from the raster only if there are valid coordinates
    if valid_indices:
        # Batch read: read all required pixels at once
        window = rasterio.windows.Window.from_slices(
            (min(px for _, px, _ in valid_indices), max(px for _, px, _ in valid_indices) + 1),
            (min(py for _, _, py in valid_indices), max(py for _, _, py in valid_indices) + 1)
        )
        # Read data as a numpy array and then transfer it to GPU
        data = src.read(1, window=window)
        data_gpu = cp.asarray(data)  # Transfer data to GPU
        
        print("complete map import")
        
        # Map the read values back to the positions in the output array
        PRCS = 0
        STEP = 10
        k = 0
        for i, px, py in valid_indices:
            # Adjust indices to window
            window_x = px - window.row_off
            window_y = py - window.col_off
            values[i] = data_gpu[window_x, window_y]  # Access the GPU array
            k += 1
            if((k*100)/len(valid_indices) > PRCS):
                PRCS += STEP
                print(f"{PRCS}% complete")
    
    return cp.asnumpy(values)

def tiffwarp_batch(DATAMAP,Prof,Range,d):
    
    # Open the raster file
    Nw = int((Range[1]-Range[0])//d)
    coords = []
    for row in Prof:
        for i in range(Nw):
            L = Range[0]+i*d
            P = (row[0] + row[2]*L, row[1] + row[3]*L)
            coords.append(P)
    print("Open the raster file")
    with rasterio.open(DATAMAP) as src:
        z_values = GetZ_batch_gpu(src, coords)
    Warped = z_values.reshape(len(Prof),Nw)
    return(Warped)

def main(DATAMAP,DATAPROFILE,d,reverse):
    #DATAMAP: geotiff DEM datafile
    #DATAPROFILE: polyline datafile for survey profile
    #d: grid size for output warped data
    #reverse=0: lefthandside reverse=1: righthandside
    
    MAPFILE = f"..//DATA//{DATAMAP}"
    PROFFILE = f"..//config//{DATAPROFILE}"
    OUTMAP = f"..//OUT//warped_{DATAMAP}"

    R,P = ReadProfileFile(PROFFILE)
    
    if len(P)<2:
        print("invalid number of points")
    elif(len(P)==2):
        Q = np.zeros((3,2))
        Q[0,:] = P[0,:]
        Q[2,:] = P[1,:]
        Q[1,:] = (P[0,:]+P[1,:])/2
        P = Q

    if len(P)==3:
        Q = np.zeros((5,2))
        Q[0,:] = P[0,:]
        Q[2,:] = P[1,:]
        Q[4,:] = P[2,:]
        Q[1,:] = (P[0,:]+P[1,:])/2
        Q[3,:] = (P[1,:]+P[2,:])/2
        P = Q

    if reverse == 1:
        P = P[::-1,:]
        
    P_new, dp = bspline(P,d)
    N = normal(P_new)
    Prof = np.column_stack([P_new, N])
    
    Warped = tiffwarp_batch(MAPFILE,Prof,R,d)
    Warped[Warped < -900] = np.nan

    # Create the transform
    transform = Affine(d, 0.00, 0.00,
                       0.00, -dp, 0.00)
        
    # Create a new raster data source with one band
    with rasterio.open(
        OUTMAP, 'w',
        driver='GTiff',
        height=Warped.shape[0],
        width=Warped.shape[1],
        count=1,
        dtype=Warped.dtype,
        crs=None,  # No coordinate reference system
        transform=transform
    ) as dst:
        dst.write(Warped, 1)  # Write data to the first band

    
if __name__ == "__main__":
    main()