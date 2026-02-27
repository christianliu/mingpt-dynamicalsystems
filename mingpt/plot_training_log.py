import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_train_test(work_dir):
    """
    Given working directory containing "train_test_log.csv"
    with the columns 'train_score', 'test_score', plot 
    """

    log_path = os.path.join(work_dir, "train_test_log.csv")
    df = pd.read_csv(log_path)
    df['total'] = df['train_score'] + df['test_score']

    ax = df[['train_score', 'test_score', 'total']].plot(
        color=['red', 'green', 'blue'], 
        grid=True,
        title="Test and training scores during training"
    )
    ax.set_xlabel("Iteration (x500)")
    ax.set_ylabel("Score")
    plt.show()


if __name__ == '__main__':
    plot_train_test("saved params/run-4")