# TerraceConnect User Manual
## Overview
**TerraceConnect.py** is a tool designed to automatically extract cliff features from DEM data of terrace landscapes and assist in determining the connectivity of synchronous terraces.

For detailed processes, please refer to **Komori et al. (in prep)**.

## 1. Preparing DEM Data
First, the DEM data needs to be reshaped into a rectangular form by warping it along a survey profile. 

Before running the code, prepare both the source DEM data and the survey profile.

### DEM Data
Prepare the DEM data in **GeoTIFF** format. Ensure that the tiff file is output in a metric coordinate system (e.g., Pseudo-Mercator or UTM coordinate). The input DEM data should be saved in `/DATA`.

### Survey Profile
Prepare the survey profile and save it as `profile.dat` in `/config`. The format of `profile.dat` should look like this:
```
Range -1000 2000
#Control Points
-7177.611   -114991.478
-6808.968   -115313.512
-6347.103   -115567.749
-5749.646   -115779.613
-5181.850   -115843.172
-4601.342   -115843.172
```
- **Range** specifies the output extent perpendicular to the survey line.
- Under **Control Points**, list the vertices of the survey profile as a polyline. It is generatable using GIS software. Ensure the coordinate system matches that of the input DEM data.

Next, run the following code as `TerraceConnect.py`:

```python
from modules import Warp
DATAPROFILE = "profile.dat"
DATAMAP = "DEM_FILENAME.tif"

d = 10 #output grid size[m]
reverse = 0 #direction of the survey profile

Warp.main(DATAMAP,DATAPROFILE,d,reverse)
```
- `d`: Specifies the output grid size.
- `reverse`: Controls the direction of the survey line. If `reverse=0`, the terrain on the left side of the line is output, and if `reverse=1`, the terrain on the right side is output.

The output geotiff file will be saved in `/Warped`.

## 2. Detecting Cliff Features and Terrace Distribution
Before running the cliff feature detection, the parameters should be configured in `params.dat` located in `/config`
```
r	10.
dz	0.7
Hmin	0
Hmax	40
Wr	600
dx	20
dw  20
mode    4
Fill    1
n_bootstrap 20
Kmin    3
Kmax    8
zint    0.5
position    900
```

- `r`, `dz`: Criteria for characteristic cliff features.
- `Hmin`, `Hmax`: Elevation range for detecting cliff features.
- `Wr`: Width of each analysis window.
- `dx`: Horizontal step size for analysis windows.
- `dw`: Interval between detection transects for cliffs.
- `mode`: Analysis mode.
- `Fill`: Whether to fill pitholes in surface elevation transects. 0: no fill, 1: fill
- `n_bootstrap`: Number of bootstrap trials.
- `Kmin`, `Kmax`: Range for exploring the number of clusters (K).
- `zint`: Vertical grid size for displaying detection probability.
- `position`: Specifies the x-location when running modes 1, 2, or 3.

### Analysis Modes
To properly run **TerraceConnect**, parameter tuning is necessary. By switching between analysis modes, you can output results such as cliff detection on a single elevation transect, GMM clustering results in a specific analysis window, or bootstrap results. Explore the optimal parameter settings by referring to these results.

#### `mode=1`: Cliff Detection on a Single Elevation Transect
This mode displays the locations of cliffs detected on the transect at the x-location specified by `position` under the criteria set in `params`.
    
#### `mode=2`: GMM Clustering in a specified Analysis Window
This mode outputs the vertical histogram of the extracted cliff features from the selected analysis window and the GMM approximation with the optimal number of clusters determined by the AIC. The window's position is centered on the x-location specified by `position`.
    
#### `mode=3`: Bootstrap Result in a specified Analysis Window
This mode outputs the histogram of the bootstrapping result from the selected analysis window.
    
#### `mode=4`: Bootstrap Evaluation Across the Entire Section
This mode performs bootstrap analysis across the entire warped DEM and outputs the detection probability.
