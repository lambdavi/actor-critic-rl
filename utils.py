import matplotlib.pyplot as plt
def plot_learning_curve(x, y, path) -> None:
    plt.style.use('seaborn')
    plt.plot(x,y)
    plt.savefig(path)