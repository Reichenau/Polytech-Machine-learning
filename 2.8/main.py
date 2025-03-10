import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# ----------------------------- График № 1 -----------------------------

rng = np.random.default_rng(1)

x = np.linspace(0, 20, 5)

y1 = rng.integers(0, 12, 5)
y2 = rng.integers(0, 12, 5)

plt.plot(x, y1, "o-r", label="line 1")
plt.plot(x, y2, "o-.g", label="line 1")


plt.legend(loc="upper left")
plt.show()

# ----------------------------- График № 2 -----------------------------

grid = plt.GridSpec(2, 2)

rng = np.random.default_rng(3)

x = np.linspace(1, 5, 5)
y1 = rng.integers(0, 5, 5)
y2 = (x - 3) ** 2 + 4 
y3 = -(x - 3) ** 2 + 2

plt.subplot(grid[0, :2]).plot(x, y1)
plt.xticks(np.arange(1, 5.5, 0.5))

plt.subplot(grid[1, :1]).plot(x, y2)
plt.xticks(np.arange(1, 5.5, 0.5))

plt.subplot(grid[1, 1:]).plot(x, y3)
plt.xticks(np.arange(1, 5.5, 0.5))

plt.savefig("img.png")
plt.show()

# ----------------------------- График № 3 -----------------------------

x = np.linspace(-5, 5, 11)
y = x ** 2

plt.plot(x, y)
plt.annotate("min", xy=(0, 0), xytext=(0, 10), arrowprops=dict(facecolor="green"))
plt.show()

# ----------------------------- График № 4 -----------------------------

data = np.random.randint(11, size=(7, 7))

fig, ax = plt.subplots()
dt = ax.pcolor(data)

# Как опустить colorbar вниз я не знаю, поэтому пришлось подсмотреть
axins = inset_axes(ax,
                   width="7%",
                   height="50%",
                   loc='lower left',
                   bbox_to_anchor=(1.02, 0., 1, 1),
                   bbox_transform=ax.transAxes,
                   )
plt.colorbar(dt, cax=axins)

plt.show()

# ----------------------------- График № 5 -----------------------------

fig, ax = plt.subplots()

x = np.linspace(0, 5, 1000)
y = np.cos(np.pi * x)

ax.plot(x, y, "r")
ax.fill_between(x, y, color="blue", alpha=0.7)
plt.show()

# ----------------------------- График № 6 -----------------------------

x = np.linspace(0, 5, 1000)
y = np.cos(np.pi * x)


plt.plot(x, [i if (i >= -0.5) and (i <= 1) else np.nan for i in y], linewidth=3)
plt.ylim(-1, 1)
plt.show()

# ----------------------------- График № 7 -----------------------------

fig, ax = plt.subplots(1, 3, figsize=(13, 4))

x = np.linspace(0, 6, 6)
y = x

style = ["pre", "post", "mid"]

for item, gr in enumerate(ax):
    gr.step(x, y, "g-o", where=style[item])
    gr.grid()

plt.show()

# ----------------------------- График № 8 -----------------------------

x = np.linspace(0, 10, 100)

y1 = np.sin(x)
y2 = 2 * np.sin(x)
y3 = 4 * np.sin(x)

plt.stackplot(x, y1, y2, y3, labels=["$y_1$", "$y_2$", "$y_3$"])

plt.legend(loc='upper left')
plt.show()

# ----------------------------- График № 9 -----------------------------

data = np.random.randint(1, 50, 5)
marks = ["BMW", "Toyota", "Ford", "Jaguar", "Audi"]

plt.pie(data, labels=marks, explode=(0.1, 0, 0, 0, 0))

plt.show()

# ----------------------------- График № 10 ----------------------------

data = np.random.randint(1, 50, 5)
marks = ["BMW", "Toyota", "Ford", "Jaguar", "Audi"]

plt.pie(data, labels=marks, wedgeprops={'width': 0.3})

plt.show()