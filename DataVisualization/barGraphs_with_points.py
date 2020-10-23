import matplotlib.pyplot as plt
np.random.seed(123)

w = 0.8    # bar width
x = [1, 2] # x-coordinates of your bars
colors = [(0, 0, 1, 1), (1, 0, 0, 1)]    # corresponding colors

y = np.array([np.array([56.00,23.00,50.00]),np.array([86.00,103.00,137.00,70.00])])   #data series -- manually input values into each array 

fig, ax = plt.subplots()
ax.bar(x,
       height=[np.mean(yi) for yi in y],
       yerr=[np.std(yi) for yi in y],    # error bars
       capsize=12, # error bar cap width in points
       width=w,    # bar width
       tick_label=["NREM high delta subgroup", "NREM low delta subgroup"],
       color=(0,0,0,0),  # face color transparent
       edgecolor=colors,
       #ecolor=colors,    # error bar colors; setting this raises an error for whatever reason.
       )

for i in range(len(x)):
    # distribute scatter randomly across whole width of bar
    ax.scatter(x[i] + np.random.random(y[i].size) * w - w / 2, y[i], color=colors[i])

plt.show()