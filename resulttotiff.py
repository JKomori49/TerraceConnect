from http.client import NETWORK_AUTHENTICATION_REQUIRED
import numpy as np
import rasterio
from rasterio.transform import from_origin

def main():
    data = np.loadtxt("../OUT/Sunosakitest00.txt")
    Nx, Ny = data.shape
    
    data_T = np.zeros((Ny,Nx))
    for i in range(Ny):
        for j in range(Nx):
            data_T[i,j] = data[j,Ny-i-1]
    
    dx = 20
    H = 40
    ampl = 40
    Z0 = 0000
    
    n_bootstrap = 100
    
    data_T /= n_bootstrap

    Ny_S = int(Ny/4)
    data_S = np.zeros((Ny_S,Nx))
    for i in range (0,Ny,4):
        data_S[round(i/4),:] = data_T[i,:]+data_T[i+1,:]+data_T[i+2,:]+data_T[i+3,:]


    Lx = Nx*dx
    Ly = H*ampl
    Oy = Z0
    
    pixel_height = Ly / Ny_S
    pixel_width = Lx / Nx

    transform = from_origin(0, Oy, pixel_width, pixel_height)

    metadata = {
        'driver': 'GTiff',
        'dtype': 'float32',  # Change dtype if your data is integers
        'nodata': None,
        'width': Nx,
        'height': Ny_S,
        'count': 1,
        'crs': None,  # No coordinate reference system
        'transform': transform
    }

    # Step 3: Write the array to a TIFF file
    with rasterio.open('output.tif', 'w', **metadata) as dst:
        dst.write(data_S, 1)  # Write to the first band



if __name__ == "__main__":
    main()