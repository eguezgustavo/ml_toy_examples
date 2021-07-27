import math
from matplotlib import gridspec
from matplotlib import pyplot
from mlxtend import plotting


def plot_multiple_predictions(supported_vector_classifiers, titles, accuracies, x_test, y_test):
    number_of_columns = 3
    number_of_rows = math.ceil(len(supported_vector_classifiers) / number_of_columns)
    figure = pyplot.figure(constrained_layout=True, figsize=(20, 6 * number_of_rows))
    grid_specification = gridspec.GridSpec(number_of_rows, number_of_columns, figure=figure)
    plots = [
        figure.add_subplot(grid_specification[row, column])
        for row in range(number_of_rows) for column in range(number_of_columns)
    ]
    
    for index, plot in enumerate(plots):
        pyplot.sca(plot)
        plot.set_title(f'{titles[index]} (Accuracy: {accuracies[index]})')
        plot.set_xlabel('Feature 1')
        plot.set_ylabel('Feature 2')
        _ = plotting.plot_decision_regions(x_test, y_test, clf=supported_vector_classifiers[index])
