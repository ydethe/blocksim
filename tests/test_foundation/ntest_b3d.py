from blocksim.graphics.BFigure import FigureFactory
from blocksim.graphics.GraphicSpec import FigureProjection
from blocksim.graphics import showFigures


def ntest_b3d():
    fig = FigureFactory.create(title="Panda 3D", projection=FigureProjection.EARTH3D)
    gs = fig.add_gridspec(1, 1)
    fig.add_baxe(title="", spec=gs[0, 0])
    showFigures()


if __name__ == "__main__":
    ntest_b3d()
