import pandas as pd
import numpy as np
import statistics
import copy
import math
from scipy.optimize import curve_fit
import visualize

# Calculate root-mean-square error
def rmse(v1, v2):
    sum = 0
    for i in range(len(v1)):
        sum += (v1[i] - v2[i]) ** 2
    return (sum / len(v1)) ** (1/2)

# Calculate root-mean-square percentage error
def rmspe(y_true, y_pred):
    sum = 0
    for i in range(len(y_true)):
        sum += ((y_true[i] - y_pred[i]) / y_true[i]) ** 2
    return (sum / len(y_true)) ** (1/2) * 100

# Calculate mean absolute percentage error
def mape(y_true, y_pred):
    sum = 0
    for i in range(len(y_true)):
        sum += abs(y_true[i] - y_pred[i]) / y_true[i]
    return sum / len(y_true) * 100

# Calculate r-squared value
# Inputs: function, x_real, y_real
def calculate_r_squared(p, x, y):
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((y - yhat)**2)
    sstot = np.sum((y - ybar)**2)
    rsq = 1 - (ssreg / sstot)
    return rsq

# Calculate r-squared value
# Inputs: y_pred, y_real
def calculate_r_squared_2(yhat, y):
    yhat = np.array(yhat)
    y = np.array(y)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((y - yhat)**2)
    sstot = np.sum((y - ybar)**2)
    rsq = 1 - (ssreg / sstot)
    return rsq

# Linear function for curve fitting
def linear_func(x, a, b):
    return a*x + b

# Quadratic function for curve fitting
def quadratic_func(x, a, b, c):
    return a*x**2 + b*x + c

# Combined cube-root and linear function for curvefitting
def lin_cbrt_func(x, a, b, c):
    return a*x + b*x**(1/3) + c

# Convert thicknesses from angstroms to nm in raw data
def raw_thickness_to_nm(df, save_path):
    mws = ['30K','50K','123K','200K','311K','650K','1080K','2000K']

    for mw in mws:
        vals = list(df[mw])
        vals = [v / 10 for v in vals]
        df[mw] = vals

    df.to_csv(save_path, index=False)

# Convert thicknesses from angstroms to nm in test data
def test_thickness_to_nm(df, save_path):
    mws = ['25K','25K_2','290K','709K','709K_2','280K PD','280K PD_2']

    for i in range(len(df)):
        if 'Thickness' in str(df.at[i, 'mW']) or 'Averages' in str(df.at[i, 'mW']):
            for mw in mws:
                df.at[i, mw] = df.at[i, mw] / 10

    df.to_csv(save_path, index=False)

# Average the thickness measurements for each concentration and molecular weight
def raw_to_average(df, save_path):
    concentrations = [10, 15, 20, 25, 30]
    mws = ['30K','50K','123K','200K','311K','650K','1080K','2000K']

    # Store values for dataframe
    all_avgs = []
    # Store values for stdev calculation
    all_stdevs = []
    all_actuals = []
    all_means = []
    for mw in mws:
        # Create dictionary
        concentration_trials = dict()
        for c in concentrations:
            concentration_trials[c]=[]
        
        # Add trials to dict
        for i in range(len(df)):
            c = df.at[i, 'Concentration']
            if c in concentrations:
                concentration_trials[c].append(df.at[i, mw])
        
        # Average trials
        avgs = []
        for c in concentrations:
            avgs.append(statistics.mean(concentration_trials[c]))
        all_avgs.append(avgs)

        # Find standard deviations
        stdevs = []
        for c in concentrations:
            # 95% CI
            stdevs.append(statistics.stdev(concentration_trials[c]) * 1.96 / math.sqrt(len(concentration_trials[c])))
        all_stdevs.append(stdevs)

        for i, c in enumerate(concentrations):
            avg = all_avgs[-1][i]
            all_actuals.extend(concentration_trials[c])
            all_means.extend([avg for _ in range(len(concentration_trials[c]))])

    # Save to df
    new_df = pd.DataFrame()
    new_df['Concentration'] = concentrations
    for i, mw in enumerate(mws):
        new_df[mw] = all_avgs[i]
        new_df[mw + ' Err'] = all_stdevs[i]
    new_df.to_csv(save_path, index=False)

# Calculate critical concentration points
# These points are required to train the manifold
def find_critical_points(df, save_folder, save=True):
    mws = ['30K','50K','123K','200K','311K','650K','1080K','2000K']
    c_thicknesses = [100,200,300,400,500]
    concentrations = [10, 15, 20, 25, 30]

    # Dataframe for storing results
    results = pd.DataFrame()
    results['Thickness'] = c_thicknesses
    
    # Store values for plotting
    all_concentrations = []
    all_thicknesses = []
    all_curves = []
    all_rsq = []
    all_err_bars = []
    all_err_bars_cc = []

    all_coeffs = []

    for mw in mws:
        cur_ccs = []

        # Get datapoints
        concentrations = []  # xval
        thicknesses = []  # yval
        err_bars = []  # error bars
        for i in range(len(df)):
            concentrations.append(df.at[i, 'Concentration'])
            thicknesses.append(df.at[i, mw])
            err_bars.append(df.at[i, mw+' Err'])
        all_concentrations.append(concentrations)
        all_thicknesses.append(thicknesses)
        all_err_bars.append(err_bars)
        
        # Curvefit - polynomial
        coeffs = np.polyfit(concentrations, thicknesses, 2)
        p = np.poly1d(coeffs)
        all_coeffs.append(coeffs)
        all_curves.append(p)

        # Calculate roots - critical concentrations
        for t in c_thicknesses:
            c = copy.deepcopy(coeffs)
            c[-1] -= t
            p_t = np.poly1d(c)
            roots = p_t.r
            cur_ccs.append(max(roots))

        # Calculate error (95% CI) of critical concentrations
        n_trials = 1000
        trials = [[] for _ in range(len(c_thicknesses))]
        for _ in range(n_trials):
            t_err = [thicknesses[i] + err_bars[i]*np.random.normal() for i in range(len(thicknesses))]
            coeffs_err = np.polyfit(concentrations, t_err, 2)

            for i, t in enumerate(c_thicknesses):
                c = copy.deepcopy(coeffs_err)
                c[-1] -= t
                p_err = np.poly1d(c)
                root = max(p_err.r)
                if not isinstance(root, complex):
                    trials[i].append(root)
        err_bars_cc = [statistics.stdev(trials[i]) * 1.96 / len(trials[i]) for i in range(len(trials))]
        all_err_bars_cc.append(err_bars_cc)

        # Calculate r-squared
        # Formula: r-squared = 1 - [(y-yhat)^2 / (y-ybar)^2]
        x = concentrations
        y = thicknesses
        rsq = calculate_r_squared(p, x, y)
        all_rsq.append(rsq)

        # Visualize
        if save:
            visualize.vis_thickness_vs_concentration(mw, concentrations, thicknesses, err_bars, err_bars_cc, cur_ccs, c_thicknesses, p, coeffs, rsq, save_folder)

        # Save results
        results[mw] = cur_ccs

    # Save
    if save:
        visualize.vis_combined_thickness_vs_concentration(mws, all_concentrations, all_thicknesses, all_err_bars, all_curves, all_rsq, save_folder)
        results.to_csv(save_folder + 'critical_concentrations.csv', index=False)

    return all_coeffs

# Calculate percentage difference in thickness after annealing
def annealed_percent_difference(df, save_path, save_path_figure):
    concentrations = [10,15,20,25,30]
    mws = ['50K', '311K', '650K']
    df.set_index(keys='MW', inplace=True)

    save_df = pd.DataFrame()
    save_df['MW'] = mws
    for c in concentrations:
        percentages = []
        for mw in mws:
            og = df.at[mw, str(c)]
            annealed = df.at[mw + ' Annealed', str(c)]
            perc_diff = (og - annealed) / og * 100
            percentages.append(perc_diff)
        save_df[c] = percentages
    save_df.to_csv(save_path)

    # Visualize
    visualize.vis_annealed_percentage_difference_line(save_df, save_path_figure, 'dashed')

# Calculate relationship between Thickness and Molecular Weight
def thickness_vs_mw(df, save_folder):
    mw_names = ['30K','50K','123K','200K','311K','650K','1080K','2000K']
    mws = [30000, 50000, 123000, 200000, 311000, 650000, 1080000, 2000000]  # xvals
    concentrations = list(df['Concentration'])
    df.set_index(keys='Concentration', inplace=True)

    all_thicknesses = []
    all_err_bars = []
    all_rsq = []
    all_p = []
    all_yhat = []
    for c in concentrations:
        thicknesses = []  # yvals
        err_bars = []  # error bars
        for mw in mw_names:
            thicknesses.append(df.at[c, mw])
            err_bars.append(df.at[c, mw + ' Err'])

        # Fit
        # Cube root + linear
        coeffs, _ = curve_fit(lin_cbrt_func, mws, thicknesses)
        p = None

        # Calculate r-squared
        yhat = []
        for mw in mws:
            yhat.append(lin_cbrt_func(mw, coeffs[0], coeffs[1], coeffs[2]))
        rsq = calculate_r_squared_2(yhat, thicknesses)

        # Plot
        visualize.vis_thickness_vs_mw(mws, thicknesses, err_bars, yhat, coeffs, rsq, c, save_folder)

        all_thicknesses.append(thicknesses)
        all_err_bars.append(err_bars)
        all_rsq.append(rsq)
        all_p.append(p)
        all_yhat.append(yhat)
    
    # Plot
    visualize.vis_combined_thickness_vs_mw(mws, all_thicknesses, all_err_bars, all_p, all_yhat, all_rsq, concentrations, save_folder)

if __name__ == '__main__':
    df_raw = pd.read_csv('Data/Angstroms/raw_data.csv')
    df_test = pd.read_csv('Data/Angstroms/test_data.csv')
    df_raw_nm = pd.read_csv('Data/raw_data_nm.csv')
    df_test_nm = pd.read_csv('Data/test_data_nm.csv')
    df_avg = pd.read_csv('Data/averaged.csv')

    #raw_thickness_to_nm(df_raw, 'Data/raw_data_nm.csv')
    #test_thickness_to_nm(df_test, 'Data/test_data_nm.csv')
    #raw_to_average(df_raw_nm, 'Data/averaged.csv')
    #find_critical_points(df_avg, 'Critical Concentrations/')
    #thickness_vs_mw(df_avg, 'Thickness vs MW/')