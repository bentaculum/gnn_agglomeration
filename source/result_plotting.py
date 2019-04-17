import torch
import chartify
import pandas as pd
import numpy as np
import os


class ResultPlotting:
    def __init__(self, config):
        self.config = config

    def plot_errors_by_location(self, data, predictions, targets):
        if not self.config.plot_error_locations:
            return

        all_x_pos = []
        all_y_pos = []
        for d in data:
            all_x_pos.extend(d.pos[:, 0].tolist())
            all_y_pos.extend(d.pos[:, 1].tolist())

        ch = chartify.Chart(blank_labels=True)
        error = np.array(predictions) - np.array(targets)
        df = pd.DataFrame({'x': all_x_pos, 'y': all_y_pos, 'error': error})
        df = df[df['error'] != 0]

        ch.plot.text(
            data_frame=df,
            x_column='x',
            y_column='y',
            text_column='error'
        ).axes.set_xaxis_label('x') \
            .axes.set_yaxis_label('y') \
            .set_title('Errors by location in euclidian space') \
            .set_subtitle('Each number corresponds to prediction-target for a misclassified node')
        ch.save(filename=os.path.join(self.config.temp_dir, 'errors_by_location.png'), format='png')
