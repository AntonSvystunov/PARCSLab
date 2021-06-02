import matplotlib.pyplot as plt

class TestInfo:
    def __init__(self, label, n, duration):
        self.label = label
        self.n = n
        self.duration = duration

def show_plot(test_infos, xlabel = 'x', ylabel = 'duration (seconds)'):
    lines_dict = {}
    for result_set in test_infos:
        for test_case in result_set:
            key = test_case.label
            x, y = test_case.n, test_case.duration
            if key not in lines_dict:
                lines_dict[key] = [[x],[y]]
            else:
                lines_dict[key][0].append(x)
                lines_dict[key][1].append(y)

    for key in lines_dict.keys():
        plt.plot(lines_dict[key][0], lines_dict[key][1], label = key)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend()
    plt.show()





