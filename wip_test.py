import numpy as np
from numpy import pi
from blocksim.dsp.DSPSignal import DSPSignal
from wip import LineFactory, AxeFactory, FigureFactory, render, showFigures


sig = DSPSignal.fromLinearFM(
    name="sig", samplingPeriod=1e-2, samplingStart=0, tau=50e-2, fstart=-20, fend=20
)

# # =========================
# # 3D Earth plot
# # =========================
# fig = FigureFactory.create(title="Figure", projection="earth3d")
# gs = fig.add_gridspec(1, 1)

# axe = AxeFactory.create(spec=gs[0, 0], title="axe", projection="surface")

# axe.plot(line=sig, color="red")

# render(fig)

# =========================
# 2D Earth plot
# =========================
# fig = FigureFactory.create(title="Figure")
# gs = fig.add_gridspec(1, 1)

# axe = AxeFactory.create(spec=gs[0, 0], title="axe", projection="PlateCarree")

# # axe.plot(line=sig, color="red")
# lon=np.linspace(-pi,pi,50)
# lat=np.cos(lon)
# axe.plot(line=(lon,lat), color="red")

# render(fig)

# =========================
# 2D polar plot
# =========================
fig = FigureFactory.create(title="Figure")
gs = fig.add_gridspec(1, 1)

axe = AxeFactory.create(spec=gs[0, 0], title="axe", projection="north_polar")

theta = np.linspace(0, 2 * pi, 50)
r = theta / (2 * pi)
axe.plot(line=(theta, r), color="red")

render(fig)

showFigures()
