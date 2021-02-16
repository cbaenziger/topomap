#!/usr/bin/env python3
# 
#  Copyright 2021 Clay Baenziger
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# 

import copy
import csv
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from collections import defaultdict
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors

# read in data
heights = defaultdict(dict)
with open("./Grading/from_datum.csv") as f:
    h = csv.DictReader(f, fieldnames=["x", "y", "z"])
    for l in h:
        heights[int(l['x'])].update({int(l['y']):float(l['z'])})

corrections = dict()
with open("./Grading/corrections.csv") as f:
    c = csv.DictReader(f, quoting=csv.QUOTE_NONNUMERIC,
                       fieldnames=["site", "eye piece", "nw site"])
    # skip header
    next(c)
    for r in c:
        corrections[int(r["site"])] = {"eye piece": r["eye piece"],
                                       "nw site": r["nw site"]}

# normalize heights to datum height
for row in corrections.keys():
    datum_adjustment = heights[row][0]-corrections[row]["nw site"]
    eye_adjustment = corrections[row]["eye piece"]
    for y in (y for y in heights[row].keys() if (y > 0)):
        heights[row][y] = heights[row][y] - datum_adjustment
        heights[row][y] = heights[row][y] - eye_adjustment
base_datum_eye_adjustment = corrections[0]["eye piece"]
for x in heights.keys():
    for y in (y for y in heights[x].keys() if y == 0 and x > 0):
        heights[x][y] = heights[x][y] - base_datum_eye_adjustment


# create data arrays
x_array = list()
y_array = list()
z_array = list()
x_keys = list(heights.keys())
x_keys.sort()
for (x, d) in ((x, heights[x]) for x in x_keys):
     for (y, z) in d.items():
         x_array.append(x)
         y_array.append(y)
         z_array.append(z)

# build a scatter plot of observations
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title("Builders-Level Observations of Terrain Surface", fontsize=16)
ax.set_xlabel('North-South distance (ft)')
ax.set_ylabel('East-West distance (ft)')
ax.set_zlabel('Depth from Datum')
ax.scatter([-1*v for v in x_array], y_array, z_array)

# Filled plots showing terrain surface
def cc(arg):
    ''' a helper color function '''
    return mcolors.to_rgba(arg, alpha=0.6)

# polygon vertecies used for filled plot graph
# build a polygon along the x-axis
x_verts=[]
# build a polygon along the y-axis for all y=0 observations
y_verts=[]
x_keys = list(heights.keys())
x_keys.sort()
for (x, d) in ((x, heights[x]) for x in x_keys):
    # build a polygon along the x-axis
    x_line = []
    for (y, z) in d.items():
        if y == 0:
          # for the y-axes polygon need x, z points
          y_verts.append((-1*x,z))
        # for x-axes polygons need y, z points
        x_line.append((y,z))
    # only plot filled x-axes lines if we have two or more observations
    if len(x_line) > 1:
      # set the last value of every filled plot line to 0
      x_line.append((y+1,0))
      x_verts.append(x_line)

# set the last value of the y-axis plot line to 0
y_verts.append((-0.01,min((a[1] for a in y_verts))))
# need a 2D array in the end (we only have one polygon)
y_verts = [y_verts]

poly = PolyCollection(x_verts, facecolors=[cc('r'), cc('b'), cc('y')])
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=[-1*a for a in x_keys if len(heights[a])>1], zdir='x')
ax.set_xlim3d(-1*max(x_keys),min(x_keys))
ax.set_ylim3d(min(y_array), max(y_array))
ax.set_zlim3d(min(z_array), max(z_array))

poly = PolyCollection(y_verts, facecolors=[cc('g')])
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=[0], zdir='y')
ax.set_xlim3d(-1*max(x_keys),min(x_keys))
ax.set_ylim3d(min(y_array), max(y_array))
plt.show()

# calculate numerical integration of fill needed

# use the observed values making each slice as thick
# as the space to the next observation slice
# hold tuple of (x_val, integral)
x_integrals=[]
for k in x_keys:
  y_keys = list(heights[k].keys())
  y_keys.sort
  # zero measures that are cut slopes (negative fills)
  non_neg_values = [max(heights[k][v],0) for v in y_keys]
  one_d_integration = np.trapz(y_keys, non_neg_values)
  # only record integrations and x values that are non-zero
  if one_d_integration > 0:
      x_integrals.append((k, one_d_integration))

# find distance between x-rows
x_rows = [x_val for (x_val,integral) in x_integrals]
row_widths = list(np.diff(x_rows))
# get distance to last observation row
row_widths.append(max(x_keys) - max(x_rows))

# number of cubic yards of fill needed (note 27 ft per cu yd)
cubic_yards = sum(np.array(row_widths) * [integral for (x_val,integral) in x_integrals])/27
# with commas
cu_yrds_fmt = "{:,}".format(int(cubic_yards.round()))


# make regular grid for interpolation and contour plots
x_keys = list(heights.keys())
x_keys.sort()
x_grid = np.arange(0, max(x_array), 1)
y_grid = np.arange(0, max(y_array), 1)
# NOTE: z_grid is unused, but useful for play
z_grid = []
for y in y_grid:
  line = []
  for x in x_grid:
    # if we do not have a value use NaN
    # also dance around doing a look up on keys that don't exist
    if x in heights:
        line.append(heights[x][y] if y in heights[x] else np.nan)
    else:
        line.append(np.nan)
  z_grid.append(line)

# interpolate the data between our observation lines
triang = tri.Triangulation(x_array, y_array)
#interpolator = tri.LinearTriInterpolator(triang, z_array)
interpolator = tri.CubicTriInterpolator(triang, z_array, kind='min_E', trifinder=None, dz=None)
Xi, Yi = np.meshgrid(x_grid, y_grid)
zi = interpolator(Xi, Yi)

# calculate the fill needed using interpolated data
# NOTE: this uses the bounds of the x and y grid 
# but the interpolation does not go the full bounds
cubic_yards_interpolated = sum([np.trapz([j for j in i if not np.isnan(j)]) for i in zi.data])/27
# with commas
cu_yrds_fmt_interp = "{:,}".format(int(cubic_yards_interpolated.round()))

# plot contour surface
fig = plt.figure()
fig.suptitle(f"Contour Surface of {cu_yrds_fmt_interp}cu. yrds.\nFill(Cut) Needed for Level", fontsize=16)
plt.xlabel('North-South distance (ft)')
plt.ylabel('East-West distance (ft)')
levels = 15
cs = plt.contourf(-1*x_grid, y_grid, zi, levels)
cbar = fig.colorbar(cs, format='%1.1f ft')
plt.show()

# plot contour plot (2D topomap)
fig = plt.figure()
fig.suptitle(f"Contour Plot of {cu_yrds_fmt_interp}cu. yrds.\nFill(Cut) Needed for Level", fontsize=16)
levels = 10
cs = plt.contour(-1*x_grid, y_grid, zi, levels, linewidths=0.5)
plt.xlabel('North-South distance (ft)')
plt.ylabel('East-West distance (ft)')
plt.clabel(cs, cs.levels[:levels], fmt='%1.1f ft', inline=True, fontsize=10)
plt.show()

'''
References:
* Contour plot labels: https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/contour_label_demo.html
* Contourf demo: https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/contourf_demo.html#sphx-glr-gallery-images-contours-and-fields-contourf-demo-py
* More Contour Examples: https://mse.redwoods.edu/darnold/math50c/python/contours/index.xhtml
* Contour Plot of Irregularly Spaced Data (Interpolation): https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/irregulardatagrid.html
* Matplotlib Tri Docs: https://matplotlib.org/3.1.1/api/tri_api.html#matplotlib.tri.LinearTriInterpolator
* Matplotlib Polygon Plot Demo: https://matplotlib.org/mpl_examples/mplot3d/polys3d_demo.py
* Titles of graphs: https://matplotlib.org/3.1.1/gallery/subplots_axes_and_figures/figure_title.html
'''
