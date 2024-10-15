# TerraceConnect User Manual
## Overview
**TerraceConnect.py** is a tool designed to automatically extract cliff features from DEM data of terrace landscapes and assist in determining the connectivity of synchronous terraces.

For detailed processes, please refer to **Komori et al. (in prep)**.

## 1. Preparing DEM Data
First, the DEM data needs to be reshaped into a rectangular form by warping it along a survey profile. 

Before running the code, prepare both the source DEM data and the survey profile.

### DEM Data
Prepare the DEM data in **GeoTIFF** format. Ensure that the tiff file is output in a meter-based coordinate system (e.g., Pseudo-Mercator or UTM coordinate). The input DEM data should be saved in the "DATA" directory.

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
- **Range** specifies the output range perpendicular to the survey line.
- Under **Control Points**, list the vertices of the survey profile as a polyline. It is generatable using GIS software. Ensure the coordinate system matches that of the input DEM data.
