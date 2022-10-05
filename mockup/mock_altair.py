import altair as alt
from vega_datasets import data


def generate_json_spec():
    cars = data.cars()

    chart = (
        alt.Chart(cars)
        .mark_point()
        .encode(
            x="Horsepower",
            y="Miles_per_Gallon",
            color="Origin",
        )
        .interactive()
    )

    chart.save("chart.json")


def plot_from_json_spec():
    with open("chart.json", "r") as f:
        json_string = f.read()

    chart = alt.Chart.from_json(json_string, validate=True)

    # chart.show()
    chart.save("filename.html")


generate_json_spec()
plot_from_json_spec()
