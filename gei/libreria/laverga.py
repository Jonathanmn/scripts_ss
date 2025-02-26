from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, BoxSelectTool, LassoSelectTool
from bokeh.io import push_notebook
import pandas as pd
import numpy as np

from picarro import *

folder_path = '/home/jmn/picarro_data/minuto/2024/02'
gei = read_raw_gei_folder(folder_path, 'Time')

gei['Time'] = pd.to_datetime(gei['Time'])


def interactive_plot_bokeh(df):
    output_notebook()

    source = ColumnDataSource(df)

    p = figure(width=800, height=400, x_axis_type='datetime', title="Interactive Plot")
    p.circle('Time', 'CO2_Avg', source=source, size=8, color='navy', alpha=0.5)

    p.add_tools(BoxSelectTool(), LassoSelectTool())

    def update(attr, old, new):
        selected_indices = source.selected.indices
        df.loc[selected_indices, 'CO2_Avg'] = np.nan
        source.data = ColumnDataSource.from_df(df)
        push_notebook()

    source.selected.on_change('indices', update)

    show(p, notebook_handle=True)
    return df

# Example usage
updated_gei = interactive_plot_bokeh(gei)