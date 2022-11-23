import plotly.express as px


fig = px.line_polar(
    r=range(0, 90, 10),
    theta=range(0, 90, 10),
    # range_theta=[0, 90],
    start_angle=0,
    direction="counterclockwise",
)

fig2 = px.line_polar(
    r=range(0, 90, 10),
    theta=range(0, 90, 10),
    # range_theta=[0, 90],
    start_angle=0,
    direction="counterclockwise",
)

# fig.write_html("file.html",include_plotlyjs='cdn')
fig.show()
fig2.show()
