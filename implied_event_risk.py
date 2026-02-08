import os
from pathlib import Path

import matlab.engine
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import scipy.optimize as optimize
import seaborn as sns

# Engine Setup
if 'eng' not in globals():
    print("Starting MATLAB Engine...")
    eng = matlab.engine.start_matlab()
    eng.addpath(os.getcwd(), nargout=0)

def computed_implied_event_risk(P1, P2):
    
    # P1 and P2 must correspond to the same strikes and thus have the same length
    if len(P1) != len(P2):
        raise ValueError(f"Input lengths do not match: {len(P1)} vs {len(P2)}")
    
    # Ensure that the prices vector are 1D float numpy arrays
    P1 = np.array(P1, dtype=float).flatten()
    P2 = np.array(P2, dtype=float).flatten()

    N = len(P1) - 1 # number of strikes - 1

    # Compute adjusted prices
    idx_0_N = np.arange(0, N + 1)
    idx_1_Np1 = np.arange(1, N + 2)
    idx_2_Np2 = np.arange(2, N + 3)

    term1_P1 = (P1[-1] / (N + 1)) - (0.5 * (N + 2) / (N + 1))
    term1_P2 = (P2[-1] / (N + 1)) - (0.5 * (N + 2) / (N + 1))
    term2 = (idx_1_Np1 * idx_2_Np2) / (N + 1) / 2.0

    P1dot = P1 - idx_0_N * term1_P1 - term2
    P2dot = P2 - idx_0_N * term1_P2 - term2

    # Inversion step (FFT)
    shift_amount = int(-N / 2)
    B1u = np.roll(P1dot, shift_amount)
    B2u = np.roll(P2dot, shift_amount)

    # Equation 15
    Q1 = np.fft.fft(B1u)
    Q2 = np.fft.fft(B2u)

    with np.errstate(divide='ignore', invalid='ignore'):
        Div = (Q1 ** 2) / Q2
        Div[np.isnan(Div)] = 0 # set division to 0 if Q2 was equal to 0
    
    # Equation 16
    Bhat_raw = np.fft.ifft(Div)
    Bhat_real = np.real(Bhat_raw)

    # Shift back
    Bhat_adjusted = np.roll(Bhat_real, int(N / 2))

    # Recover the raw density from adjusted prices
    term_lag1 = np.roll(Bhat_adjusted, 1)
    term_lag2 = np.roll(Bhat_adjusted, 2)
    qEvent_raw = Bhat_adjusted - 2 * term_lag1 + term_lag2
    
    # Add back normalization (that had removed a quadratic term with 2nd derivative 1/(N+1))
    qEvent_raw = qEvent_raw + (1.0 / (N + 1))
    
    # Enforce boundary conditions on density to avoid numerical nosie
    qEvent_raw[0] = 0
    qEvent_raw[1] = 0
    qEvent_raw[-1] = 0
    qEvent_raw[-2] = 0
    
    # Integrate to get the full price
    PEvent_raw = np.cumsum(np.cumsum(qEvent_raw))
    
    # Convexify the full price
    PEvent_convex = project_to_convex_matlab(PEvent_raw)
    
    # Get the final density from the convexified price
    term_lag1_conv = np.roll(PEvent_convex, 1)
    term_lag2_conv = np.roll(PEvent_convex, 2)
    qEvent_final = PEvent_convex - 2 * term_lag1_conv + term_lag2_conv
    
    # Boundary cleanup
    qEvent_final[0] = 0
    qEvent_final[1] = 0
    qEvent_final[-1] = 0
    qEvent_final[-2] = 0
    
    return {
        'qEvent': qEvent_final,
        'PEvent': PEvent_convex,
        'Bhat': Bhat_adjusted # for debugging
    }



def project_to_convex_matlab(y_np):
    # Convert numpy to matlab double
    y_mat = matlab.double(y_np.tolist())
    
    # Create dummy x grid (-N/2 to N/2)
    N = len(y_np) - 1
    x_mat = matlab.double(np.arange(-N//2, N//2 + 1).tolist())
    
    # Call the script
    sp_struct = eng.ConvexSpline(
        x_mat, 
        y_mat, 
        matlab.double([]), 
        matlab.double(3), 
        matlab.double(len(y_np) - 2),                    
        matlab.double(0),      
        matlab.double([])
    )
    
    fitted_values = eng.fnval(sp_struct, x_mat)
    # Convert back
    return np.array(fitted_values).flatten()


def plot_event_risk(
    grid_strikes,
    results,
    spot_price=None,
    k_level=None,
    date_label=None,
    show_price=False,
    price_label="Recovered put price",
    dist_label="Recovered event risk distribution",
    x_label=r"$K$ (S&P index level)",
    y_label="Recovered event risk distribution",
    title="Recovered event risk distribution",
    figsize=(9.5, 5.8),
    style="whitegrid",
    palette="deep",
    fill_alpha=0.18,
    line_width=2.2,
    spot_kwargs=None,
    k_kwargs=None,
):
    """
    Plot recovered event risk distribution vs strike/index level, with optional overlays.

    Args:
        grid_strikes (np.array): x-axis values (strikes / index levels).
        results (dict): output containing 'qEvent' and optionally 'PEvent'.
        spot_price (float, optional): current spot level for a vertical marker.
        k_level (float, optional): highlight a particular strike/index level K.
        date_label (str, optional): appended to title.
        show_price (bool): if True, overlay recovered put price on right axis (requires 'PEvent').
        price_label (str): legend label for the price curve.
        dist_label (str): legend label for the distribution curve.
        x_label (str): x-axis label.
        y_label (str): left y-axis label.
        title (str): plot title (date_label appended if provided).
        figsize (tuple): figure size.
        style (str): seaborn style.
        palette (str): seaborn palette.
        fill_alpha (float): alpha for area fill under density.
        line_width (float): linewidth for density curve.
        spot_kwargs (dict, optional): overrides for spot vline styling.
        k_kwargs (dict, optional): overrides for K vline styling.

    Returns:
        (fig, ax): matplotlib figure and primary axis.
    """
    q_event = np.asarray(results["qEvent"])
    p_event = np.asarray(results["PEvent"]) if ("PEvent" in results and results["PEvent"] is not None) else None
    x = np.asarray(grid_strikes)

    if x.shape[0] != q_event.shape[0]:
        raise ValueError("grid_strikes and results['qEvent'] must have the same length.")

    if show_price and p_event is None:
        raise ValueError("show_price=True requires results['PEvent'].")

    sns.set_theme(style=style, palette=palette)
    fig, ax = plt.subplots(figsize=figsize)

    # Subtle grid, cleaner spines
    ax.grid(True, which="major", axis="both", linewidth=0.7, alpha=0.35)
    ax.grid(False, which="minor")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Main: event risk distribution
    dist_color = sns.color_palette(palette)[0]
    ax.plot(x, q_event, color=dist_color, linewidth=line_width, label=dist_label)
    ax.fill_between(x, q_event, color=dist_color, alpha=fill_alpha, linewidth=0)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Optional reference lines (spot and/or chosen K)
    spot_defaults = dict(color="black", linestyle="--", linewidth=1.4, alpha=0.75)
    k_defaults = dict(color=sns.color_palette(palette)[2], linestyle="-.", linewidth=1.6, alpha=0.9)

    if spot_kwargs:
        spot_defaults.update(spot_kwargs)
    if k_kwargs:
        k_defaults.update(k_kwargs)

    handles = []
    labels = []

    h0, l0 = ax.get_legend_handles_labels()
    handles += h0
    labels += l0

    if spot_price is not None:
        ax.axvline(spot_price, **spot_defaults)
        handles.append(plt.Line2D([0], [0], **spot_defaults))
        labels.append(f"Spot ({spot_price:,.0f})")

    if k_level is not None:
        ax.axvline(k_level, **k_defaults)
        handles.append(plt.Line2D([0], [0], **k_defaults))
        labels.append(f"$K$ ({k_level:,.0f})")

    # Optional: recovered put price on secondary axis
    ax2 = None
    if show_price:
        ax2 = ax.twinx()
        price_color = sns.color_palette(palette)[1]
        ax2.plot(x, p_event, color=price_color, linewidth=1.9, linestyle=":", label=price_label)
        ax2.set_ylabel("Option price", color=price_color)
        ax2.tick_params(axis="y", labelcolor=price_color)

        # Keep the secondary axis visually quiet
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)

        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2

    # Title
    full_title = title + (f" ({date_label})" if date_label else "")
    ax.set_title(full_title, pad=12)

    # Legend: compact and consistent
    if handles:
        ax.legend(handles, labels, frameon=False, loc="upper right")

    fig.tight_layout()
    return fig, ax


# Helper function to reduce the number of strikes
def resample_prices(strikes, prices, target_n=65):

    # Create a new uniform grid with the same range but coarser spacing
    new_strikes = np.linspace(strikes.min(), strikes.max(), target_n)

    # Interpolate prices onto new grade
    f = interp1d(strikes, prices, kind='cubic')
    new_prices = f(new_strikes)

    return new_strikes, new_prices

