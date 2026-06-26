import os
from typing import Dict, List, Tuple

import numpy as np
from matplotlib import pyplot as plt

from global_variables import DATA_DIR


def plot_alignment(
    data: Dict[str, Tuple[List[str], List[str], np.ndarray]],
    save_path: str = None,
) -> None:
    """
    Plot alignment for each pair of phrases.

    Parameters:
    - data: Dictionary containing alignment data for each pair of phrases.
            Key: Pair identifier
            Value: Tuple containing two lists of phrases and a 2D numpy array representing the alignment.
    - save_path: Optional parameter for the file name to save the figure. If None, the plot is displayed but not saved.

    Returns:
    - None
    """

    # Determine the number of rows and columns for subplots
    h, w = (len(data) + 1) // 2, 2 if len(data) % 2 == 0 else 1

    # Set the figure size based on the number of subplots
    fig_size = (10 * w, h * 10)

    # Create subplots
    _, axes = plt.subplots(h, w, figsize=fig_size)
    axes = np.array(axes)
    axes = axes.reshape(h, w)

    # Iterate through the data and plot alignment for each pair of phrases
    for i, (key, values) in enumerate(data.items()):
        phrase1, phrase2, alignment = values

        # Get the current subplot
        ax = axes[i // 2, i % 2]
        alignment_max, alignment_min = np.max(alignment, axis=1), np.min(
            alignment, axis=1
        )
        # make allignment values between 0 and 1 with 1 being the max value
        if (alignment_max - alignment_min).any():
            alignment = (alignment - alignment_min[:, None]) / (
                alignment_max[:, None] - alignment_min[:, None]
            )
        # Plot the alignment matrix
        ax.imshow(
            alignment, cmap="gray", aspect="auto"
        )  # Set aspect to "auto" to fill the entire width

        # Set ticks and labels
        ax.set_xticks(np.arange(len(phrase1)))
        ax.set_yticks(np.arange(len(phrase2)))
        ax.xaxis.tick_top()
        ax.set_xticklabels(
            phrase1, rotation=90
        )  # Rotate the text at the top by 90 degrees
        ax.set_yticklabels(phrase2)

        # Set title for the subplot
        ax.set_title(key, y=-0.1)

    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def BLEU_score_plot(data: Dict[str, np.ndarray], save_path: str = None):
    """
    Plot BLEU score for each epoch.

    Parameters:
    - data: Dictionary containing BLEU score for each epoch.
            Key: Epoch number
            Value: BLEU score
    - save_path: Optional parameter for the file name to save the figure. If None, the plot is displayed but not saved.

    Returns:
    - None
    """
    plotting_styles = ["-", "--", "-.", ":"]

    for i, (key, values) in enumerate(data.items()):
        plt.plot(values, plotting_styles[i % len(plotting_styles)], label=key)

    plt.xlabel("Sentence Length")
    plt.ylabel("BLEU score")
    plt.legend(loc="lower left")

    # Save or display the plot
    if save_path:
        save_dir = "plots"
        save_dir = os.path.join(DATA_DIR, save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, save_path)
        plt.savefig(save_file)
    else:
        plt.show()


def print_table(x: List[str], y: List[str], data: np.ndarray):
    """
    Print a table with the alignment between two phrases in a markdown format.

    Parameters:
    - x: List of words in the first phrase.
    - y: List of words in the second phrase.
    - data: 2D numpy array representing the alignment.

    Returns:
    - None
    """
    # Print the table header
    print("|", end="")
    for word in x:
        print(f" {word} |", end="")
    print()

    # Print the separator
    print("|-", end="")
    for _ in x:
        print("-|-" if _ != x[-1] else "-|", end="")
    print()

    # Print the table body
    for i, word in enumerate(y):
        print(f"| {word} |", end="")
        for j in range(len(x) - 1):
            print(f" {data[i, j]:.2f} |", end="")
        print()

    # Print the separator
    print("|-", end="")
    for _ in x:
        print("-|-" if _ != x[-1] else "-|", end="")
    print()
