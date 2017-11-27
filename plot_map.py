import bokeh
from bokeh.io import output_file, show
from matplotlib import cm
import numpy as np 
from bokeh.models import (
  GMapPlot, GMapOptions, ColumnDataSource, Circle, Square, Line, Oval, Patch, Ray, DataRange1d, PanTool, WheelZoomTool, BoxSelectTool
)


def plot_map(df, bs_latlong=None, bs=None, map_type="satellite", colorby='rssi'):
    map_options = GMapOptions(lat=.5*(df.latitude.min()+ df.latitude.max()),
                              lng=.5*(df.longitude.min()+df.longitude.max()),
                              map_type=map_type, zoom=11)

    
    plot = GMapPlot(
        x_range=DataRange1d(), y_range=DataRange1d(), map_options=map_options
    )
    plot.title.text = "Test_API_Google"

    # For GMaps to function, Google requires you obtain and enable an API key:
    #
    #     https://developers.google.com/maps/documentation/javascript/get-api-key
    #
    # Replace the value below with your personal API key:

    plot.api_key = "AIzaSyBn534zSTjx4L7l7yoklDismI0QXMiZSA8"
    
    normalize = (df[colorby] - df[colorby].min())/(df[colorby].max() - df[colorby].min())

    source = ColumnDataSource(
        data=dict(
            lat=df.latitude,
            lon=df.longitude,
            color=[cm.colors.to_hex(c) for c in cm.jet(normalize)[:,:-1]]
        )
    )


    circle = Circle(x="lon", y="lat", size=3, fill_color="color",
                       fill_alpha=0.8, line_color=None)


    plot.add_glyph(source, circle)

    if (bs is not None) & (bs_latlong is not None):
        bs = ColumnDataSource(
        data=dict(
            lat=np.array(bs_latlong[bs]['lat']).reshape(1,),
            lon=np.array(bs_latlong[bs]['long']).reshape(1,)
        )
    )


        circlebs = Square(x="lon", y="lat", size=9, fill_color="red",
                       fill_alpha=0.8, line_color=None)


        plot.add_glyph(bs, circlebs)

    plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())
    output_file("gmap_plot.html")
    show(plot)