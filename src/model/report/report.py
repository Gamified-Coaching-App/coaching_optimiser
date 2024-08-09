from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Select, Slider, TextInput, HoverTool, Div
from bokeh.layouts import column, row
import numpy as np
from utils import compute_metrics

# Load the data
REPORT_TIMESTAMP = '2024-07-31-10-51-56'
report_filepath = './reports_data/run_' + REPORT_TIMESTAMP + '.h5'
min_max_filepath = '../data/min_max_values.json'
metrics_dict, variables = compute_metrics(report_filepath, min_max_filepath)

class VariablePlots:
    def __init__(self, variable_name, metrics_dict):
        self.variable_name = variable_name
        self.metrics_dict = metrics_dict
        self.sources = {}

    def _create_scatter_source(self, epoch_key, selected_point):
        metrics = self.metrics_dict[epoch_key][self.variable_name]
        mean_actions = metrics['mean_actions']
        mean_states = metrics['mean_states']

        scatter_source = ColumnDataSource(data=dict(x=mean_states, y=mean_actions))

        selected_data = dict(x=[mean_states[selected_point]], y=[mean_actions[selected_point]]) if selected_point is not None and selected_point < len(mean_states) else dict(x=[], y=[])
        selected_source = ColumnDataSource(data=selected_data)

        return scatter_source, selected_source

    def _create_histogram_source(self, epoch_key):
        metrics = self.metrics_dict[epoch_key][self.variable_name]
        km_increase = metrics['increase']

        hist, edges = np.histogram(km_increase, bins=50)
        hist_source = ColumnDataSource(data=dict(top=hist, left=edges[:-1], right=edges[1:]))

        return hist_source

    def _create_abs_values_source(self, epoch_key, highlight_point=None):
        metrics = self.metrics_dict[epoch_key][self.variable_name]
        action_values_flat = metrics['action_values_flat']

        unique_values, counts = np.unique(action_values_flat, return_counts=True)
        y_values = np.zeros(len(action_values_flat))
        y_values[np.argsort(action_values_flat)] = np.arange(len(action_values_flat)) % counts.max()

        abs_values_source = ColumnDataSource(data=dict(
            x=action_values_flat,
            y=y_values,
            index=[f'{i % 7}.{i // 7}' for i in range(len(action_values_flat))]
        ))

        selected_data = dict(x=[], y=[], index=[])
        if highlight_point is not None and highlight_point < len(action_values_flat):
            selected_data = dict(
                x=[action_values_flat[highlight_point]],
                y=[y_values[highlight_point]],
                index=[f'{highlight_point % 7}.{highlight_point // 7}']
            )

        selected_source = ColumnDataSource(data=selected_data)

        return abs_values_source, selected_source

    def initialize_sources(self, epoch, selected_point, abs_selected_point):
        epoch_key = f'epoch_{epoch}'
        self.sources['scatter'], self.sources['scatter_selected'] = self._create_scatter_source(epoch_key, selected_point)
        self.sources['histogram'] = self._create_histogram_source(epoch_key)
        self.sources['abs_values'], self.sources['abs_selected'] = self._create_abs_values_source(epoch_key, abs_selected_point)

    def update_sources(self, epoch, selected_point, abs_selected_point):
        epoch_key = f'epoch_{epoch}'
        scatter_source, scatter_selected_source = self._create_scatter_source(epoch_key, selected_point)
        hist_source = self._create_histogram_source(epoch_key)
        abs_values_source, abs_selected_source = self._create_abs_values_source(epoch_key, abs_selected_point)

        self.sources['scatter'].data = dict(scatter_source.data)
        self.sources['scatter_selected'].data = dict(scatter_selected_source.data)
        self.sources['histogram'].data = dict(hist_source.data)
        self.sources['abs_values'].data = dict(abs_values_source.data)
        self.sources['abs_selected'].data = dict(abs_selected_source.data)

    def get_scatter_plot(self):
        scatter_plot = figure(title=f'{self.variable_name}: Actions vs States', x_axis_label='Mean States', y_axis_label='Mean Actions')
        scatter_plot.scatter('x', 'y', size=5, source=self.sources['scatter'], color="blue", legend_label='Mean KM', hover_color="orange", hover_alpha=0.5)
        scatter_plot.scatter('x', 'y', size=15, source=self.sources['scatter_selected'], color="red", legend_label='Highlighted Point', hover_color="orange", hover_alpha=0.5)
        scatter_plot.add_tools(HoverTool(tooltips=[("Index", "$index"), ("Mean States", "@x"), ("Mean Actions", "@y")], mode='mouse'))
        scatter_plot.legend.location = "top_left"
        return scatter_plot

    def get_histogram(self):
        histogram = figure(title=f'{self.variable_name}: KM Increase by Epoch', x_axis_label='% Increase', y_axis_label='Frequency')
        histogram.quad(top='top', bottom=0, left='left', right='right', source=self.sources['histogram'], fill_color="navy", line_color="white", alpha=0.5)
        return histogram

    def get_absolute_values_plot(self):
        abs_values_plot = figure(title=f'{self.variable_name}: Action Values', x_axis_label='Value', y_axis_label='Count')
        abs_values_plot.scatter('x', 'y', size=5, source=self.sources['abs_values'], color="green", legend_label='Action Values', hover_color="orange", hover_alpha=0.5)
        abs_values_plot.scatter('x', 'y', size=15, source=self.sources['abs_selected'], color="red", legend_label='Highlighted Point', hover_color="orange", hover_alpha=0.5)
        abs_values_plot.add_tools(HoverTool(tooltips=[("Index", "@index"), ("Value", "@x"), ("Count", "@y")], mode='mouse'))
        abs_values_plot.legend.location = "top_left"
        return abs_values_plot

# Bokeh Layout
variable_name = 'total km'
plots = VariablePlots(variable_name, metrics_dict)

epoch_slider = Slider(start=0, end=len(metrics_dict) - 1, value=0, step=1, title="Epoch")
point_select = TextInput(title="Select Point", value="0")
abs_point_select = TextInput(title="Select Point (Absolute Values)", value="0")

def update(attr, old, new):
    epoch = epoch_slider.value
    selected_point = int(point_select.value)
    abs_selected_point = int(abs_point_select.value)
    plots.update_sources(epoch, selected_point, abs_selected_point)

plots.initialize_sources(0, 0, 0)

scatter_plot = plots.get_scatter_plot()
histogram_plot = plots.get_histogram()
abs_values_plot = plots.get_absolute_values_plot()

epoch_slider.on_change('value', update)
point_select.on_change('value', update)
abs_point_select.on_change('value', update)

layout = column(
    row(column(Div(text="<h2>Total km</h2>"), epoch_slider, point_select, abs_point_select)),
    row(scatter_plot, histogram_plot, abs_values_plot)
)
curdoc().add_root(layout)

# Serve with Bokeh
curdoc().title = "Interactive Visualization"
