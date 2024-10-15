# TerraceConnect User Manual
## Overview
**TerraceConnect.py** is a tool designed to automatically extract cliff features from DEM data of terrace landscapes and assist in determining the connectivity of synchronous terraces.

For more detailed information on the process, please refer to **Komori et al. (in prep)**.

![](/doc/Overview.png)

## 1. Preparing DEM Data
To begin, the DEM data needs to be reshaped into a rectangular form by warping it along a survey profile. Ensure both the source DEM data and the survey profile are prepared before running the script.

### DEM Data
Prepare the DEM data in **GeoTIFF** format. Ensure that the DEM is in a metric coordinate system (e.g., Pseudo-Mercator or UTM). The input DEM data should be saved in `/DATA` directory.

### Survey Profile
Prepare the survey profile and save it as `profile.dat` in `/config` directory. The format of `profile.dat` should follow this structure:
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
- **Range**: specifies the extent perpendicular to the survey profile.
- **Control Points**: lists the vertices of the survey profile as a polyline. These can be generated using GIS software. Ensure the coordinate system matches the one used in the input DEM data.

Once the input files are ready, you can run the following code to process the DEM:

```python
from modules import Warp
DATAPROFILE = "profile.dat"
DATAMAP = "DEM_FILENAME.tif"

d = 10 #output grid size[m]
reverse = 0 #Direction of the survey profile

Warp.main(DATAMAP,DATAPROFILE,d,reverse)
```
- `d`: Specifies the output grid size in meters.
- `reverse`: Controls the direction of the survey profile.
    - `reverse=0`: Output terrain on the left side of the profile.
    - `reverse=1`: Output terrain on the right side of the profile.

The output GeoTIFF file will be saved in the `/Warped` directory.

## 2. Detecting Cliff Features and Terrace Distribution
Before running the cliff feature detection, configure the parameters in `params.dat` located in the `/config` directory.
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

- `r`, `dz`: Criteria for identifying characteristic cliff features.
- `Hmin`, `Hmax`: Elevation range for detecting cliff features.
- `Wr`: Width of the analysis window.
- `dx`: Step interval for moving the analysis windows horizontally.
- `dw`: Interval between transects for cliff extraction.
- `mode`: Specifies the analysis mode (described below).
- `Fill`: Whether to fill pitholes in surface elevation transects (0: no fill, 1: fill).
- `n_bootstrap`: Number of bootstrap trials.
- `Kmin`, `Kmax`: Range for exploring the number of clusters (K).
- `zint`: Vertical grid size for displaying detection probability.
- `position`: Specifies the x-location when running modes 1, 2, or 3.

### Analysis Modes
Once the warped DEM and input parameters are ready, you can run the following code to process the cliff detection:
```python
from modules import ExtractSteps_GMM
DATAPARAM = "params.dat"
DATAMAP = "DEM_FILENAME.tif"

ExtractSteps_GMM.main(DATAMAP,DATAPARAM)
```

Proper parameter tuning is essential for running TerraceConnect. By switching between different analysis modes, you can generate outputs like cliff detection on a single transect, GMM clustering results in a specific analysis window, or bootstrap analysis results.

#### Mode 1: Cliff Detection on a Single Elevation Transect (`mode=1`)
This mode displays the locations of cliffs detected along a transect at the x-location specified by `position`, according to the criteria set in `params.dat`.

#### Mode 2: GMM Clustering in a specified Analysis Window (`mode=2`)
This mode outputs the vertical histogram of the extracted cliff features from the selected analysis window, along with a GMM approximation. The analysis window is centered on the x-location specified by `position`.
    
#### Mode 3: Bootstrap Results in a specified Analysis Window (`mode=3`)
This mode outputs the histogram of the bootstrapping results from the selected analysis window.
    
#### Mode 4: Bootstrap Evaluation Across the Entire Section (`mode=4`)
This mode performs a bootstrap analysis across the entire warped DEM and outputs the detection probability in a color plot. The output grid data is saved to `OUT/DEM_FILENAME.dat` at the same time.
