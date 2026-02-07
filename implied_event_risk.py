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
    Bhat = np.roll(Bhat_real, int(N / 2))

    # Ensure the convexity of the price output by the FFT
    Bhat = project_to_convex_matlab(Bhat)

    # Equation 17 (recover probability)
    term_lag1 = np.roll(Bhat, 1)
    term_lag2 = np.roll(Bhat, 2)

    qEvent = Bhat - 2 * term_lag1 + term_lag2 # double differencing

    # Enforce boundary assumptions
    qEvent[0] = 0
    qEvent[1] = 0

    # Integration to recover prices
    PEvent_cdf = np.cumsum(qEvent)
    PEvent = np.cumsum(PEvent_cdf)

    return {
        'qEvent': qEvent,
        'PEvent': PEvent,
        'Bhat': Bhat
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




def plot_event_risk(grid_strikes, results, spot_price=None, date_label=None):
    """
    Plots the recovered Event Risk Density against Asset Price.
    
    Args:
        grid_strikes (np.array): The x-axis values (Asset Prices/Strikes) used for the input P1/P2.
        results (dict): The output from calculate_implied_event_risk.
        spot_price (float, optional): Current spot price to mark on the x-axis.
        date_label (str, optional): Label for the plot title.
    """
    
    q_event = results['qEvent']
    p_event = results['PEvent']
    
    # Setup aesthetic
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- Plot 1: Risk Neutral Density (Left Axis) ---
    # This corresponds to the probability mass at each asset price node
    color = 'tab:blue'
    ax1.set_xlabel('Asset Price ($S_T$)', fontsize=12)
    ax1.set_ylabel('Implied Probability Density', color=color, fontsize=12)
    
    # Plot the density as a filled area or line
    ax1.plot(grid_strikes, q_event, color=color, linewidth=2, label='Implied Event Risk Density')
    ax1.fill_between(grid_strikes, q_event, color=color, alpha=0.1)
    ax1.tick_params(axis='y', labelcolor=color)

    # Add Spot Price marker if provided
    if spot_price:
        ax1.axvline(spot_price, color='black', linestyle='--', alpha=0.7, label=f'Current Spot ({spot_price:.1f})')

    # --- Plot 2: Implied Put Price (Right Axis - Optional) ---
    # Useful to see the convexity
    ax2 = ax1.twinx()  
    color = 'tab:gray'
    ax2.set_ylabel('Implied Put Price', color=color, fontsize=12)
    ax2.plot(grid_strikes, p_event, color=color, linestyle=':', linewidth=2, label='Recovered Put Price')
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and Layout
    title = "Implied Event Risk Distribution"
    if date_label:
        title += f" ({date_label})"
    plt.title(title, fontsize=14, pad=15)
    
    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.tight_layout()
    plt.show()
    return fig, ax1


# Helper function to reduce the number of strikes
def resample_prices(strikes, prices, target_n=65):

    # Create a new uniform grid with the same range but coarser spacing
    new_strikes = np.linspace(strikes.min(), strikes.max(), target_n)

    # Interpolate prices onto new grade
    f = interp1d(strikes, prices, kind='cubic')
    new_prices = f(new_strikes)

    return new_strikes, new_prices

