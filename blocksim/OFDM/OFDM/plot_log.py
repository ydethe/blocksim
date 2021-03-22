import argparse

from parse import compile
import numpy as np
from numpy import sqrt, log10
from scipy.special import erfc
import matplotlib.pyplot as plt


def q_function(x):
    """
    https://en.wikipedia.org/wiki/Q-function
    """
    return 0.5 * erfc(x / sqrt(2))


def plot_log(fic, output=""):
    p = compile(
        "[{level}] - SNR = {snr} dB, it={it}, Bits Received = {bit_rx}, Bit errors = {bit_err}, BER = {ber}"
    )

    f = open(fic, "r")
    snr = []
    ber = []
    for line in f:
        dat = p.parse(line)
        snr.append(float(dat["snr"]))
        ber.append(float(dat["ber"]))

    c_n0 = np.array(snr) + 10 * log10(180e3)

    fig = plt.figure(dpi=150)
    axe = fig.add_subplot(111)
    axe.grid(True)
    axe.grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
    axe.semilogy(c_n0, ber, label="Simu BER")
    axe.legend()
    axe.set_xlabel("$C/N_0$ (dB)")
    axe.set_ylabel("BER")

    if output == "show":
        plt.show()
    elif output == "":
        pass
    else:
        plt.savefig(output)

    return fig


def main_plot_log():
    parser = argparse.ArgumentParser(description="Plot the result of BER computation")
    parser.add_argument("fic", help="Log file", type=str)
    args = parser.parse_args()

    plot_log(args.fic, output="show")


if __name__ == "__main__":
    main_plot_log()
