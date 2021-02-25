# This was used to generate bar graphs with each data point displayed. Used to show subgroup differences in CBS patients
# individual data point values must be specified in 'y' (line 12)

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(123)

w = 0.8    # bar width
x = [1, 1] # x-coordinates of your bars
colors = [(0, 0, 1, 1)]    # corresponding colors

y = np.array([([21.11,55.82,15.91,51.2,16.02,21.74,46.38])])   #,np.array([35.67,103.51,25.34,99.91,21.52,37.66,89.34]),np.array([21.82,35.13,7.37,21.49,10.11,10.26,21.47]),np.array([15,8.29,4.46,6.49,6.82,5.12,13.72]),np.array([2.21,1.93,1.23,2.21,3.21,2.57,3.64]),np.array([2.09,2.59,1.46,3.45,9.37,3.31,4.7]),np.array([0.53,1.11,0.44,1.8,11.13,1.44,1.09])])   #data series -- manually input values into each array 




### This works ###

w = 0.8    # bar width
x = [1] # x-coordinates of your bars
colors = [(0, 0, 1, 1)]    # corresponding colors

y = np.array([np.array([0.53,1.11,0.44,1.8,11.13,1.44,1.09])])   #data series -- manually input values into each array 

fig, ax = plt.subplots(figsize=(2,4))
ax.bar(x,
       height=[np.mean(yi) for yi in y],
       yerr=[np.std(yi)/np.sqrt(yi.shape[0]) for yi in y],    # standard error bars
       capsize=12, # error bar cap width in points
       width=w,    # bar width
       tick_label=["Gamma"],
       color=(0,0,0,0),  # face color transparent
       edgecolor=colors,
       #ecolor=colors,    # error bar colors; setting this raises an error for whatever reason.
       )

for i in range(len(x)):
    # distribute scatter randomly across whole width of bar
    ax.scatter(x[i] + np.random.random(y[i].size) * w - w / 2, y[i], color=colors[i])

plt.show()




###### This has worked before ######

w = 0.8    # bar width
x = [1, 3] # x-coordinates of your bars
colors = [(0, 0, 1, 1),(0, 0, 1, 1)]    # corresponding colors

y = np.array([np.array([21.11,55.82,15.91,51.2,16.02,21.74,46.38]),np.array([35.67,103.51,25.34,99.91,21.52,37.66,89.34])])   #data series -- manually input values into each array 

fig, ax = plt.subplots()
ax.bar(x,
       height=[np.mean(yi) for yi in y],
       yerr=[np.std(yi) for yi in y],    # error bars
       capsize=12, # error bar cap width in points
       width=w,    # bar width
       tick_label=["SO", "Delta"],
       color=(0,0,0,0),  # face color transparent
       edgecolor=colors,
       #ecolor=colors,    # error bar colors; setting this raises an error for whatever reason.
       )

for i in range(len(x)):
    # distribute scatter randomly across whole width of bar
    ax.scatter(x[i] + np.random.random(y[i].size) * w - w / 2, y[i], color=colors[i])

plt.show()