import matplotlib.pyplot as plt
import numpy as np
import copy

# Symbols for printing
one_sup = "\u00B9"
two_sup = "\u00B2"
three_sup = "\u00B3"
cbrt = "\u221B"
ang_symbol = '\u212B'

# Colors for plotting
default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Make coefficients look presentable for printing
def coeffs_for_printing(coeffs):
    print_coeffs = []

    # Round and add correct sign to coeffs
    for i in range(len(coeffs)):
        if i == 0:
            print_coeffs.append('{:.3e}'.format(coeffs[i]))
            #str(round(coeffs[i], 7)
            #"{:e}".format(0.000009732)
            continue
        if coeffs[i] < 0:
            print_coeffs.append(' - ' + str(round(abs(coeffs[i]), 3)))
        else:
            print_coeffs.append(' + ' + str(round(coeffs[i],3)))
    
    return print_coeffs

# Printing quadratic equation for Thickness-Concentration relationship
def coeffs_to_quadratic_equation(coeffs):
    print_coeffs = coeffs_for_printing(coeffs)
    equation = 'y = ' + print_coeffs[0] + '*' + 'x' + two_sup + print_coeffs[1] + '*' + 'x' + print_coeffs[2]
    return equation

# Printing log equation for Molecular Weight-Concentration relationship
def coeffs_to_log_equation(coeffs):
    print_coeffs = coeffs_for_printing(coeffs)
    equation = 'ln(y) = ' + print_coeffs[0] + '*' + 'ln(x)' + print_coeffs[1]
    return equation

# Printing combined cube-root and linear equation for Thickness-Molecular Weight relationship
def coeffs_to_cbrt_lin_equation(coeffs):
    print_coeffs = coeffs_for_printing(coeffs)
    equation = 'y = ' + print_coeffs[0] + '*x' + print_coeffs[1] + '*' + cbrt + 'x' + print_coeffs[2]
    return equation

# Visualize Thickness vs Molecular Weight relationship
# Relationship at each concentration is saved to a new figure
# Contains equations for each relationship
def vis_thickness_vs_mw(mws, thicknesses, err_bars, y_fitted, coeffs, rsq, c, save_folder):
    color = default_colors[0]

    # Plot - Line
    plt.scatter(mws, thicknesses, color=color)
    plt.plot(mws, thicknesses, color=color)
    plt.errorbar(mws, thicknesses, err_bars, capsize=7.5, linestyle='', color=color)
    plt.title(str(c) + ' mg/mL - Thickness vs Molecular Weight')
    plt.xlabel('Molecular Weight (amu)')
    plt.ylabel('Thickness (nm)')
    plt.xticks([400000*i for i in range(6)])
    plt.ticklabel_format(style='plain')
    plt.tight_layout()
    plt.savefig(save_folder + 'concentration_' + str(c) + '.png')
    plt.close()

    # Plot - Fitted Line
    # Includes equation
    plt.scatter(mws, thicknesses, label='Experimental Data', color=color)
    plt.plot(mws, y_fitted, label='Fit', color=color)
    plt.errorbar(mws, thicknesses, err_bars, capsize=7.5, linestyle='', color=color)
    plt.suptitle(str(c) + ' mg/mL - Thickness vs Molecular Weight')
    equation = coeffs_to_cbrt_lin_equation(coeffs)
    plt.title(equation + '    R' + two_sup + ': ' + str(round(rsq, 3)))
    plt.xlabel('Molecular Weight (amu)')
    plt.ylabel('Thickness (nm)')
    plt.xticks([400000*i for i in range(6)])
    plt.ticklabel_format(style='plain')
    plt.tight_layout()
    plt.savefig(save_folder + 'concentration_' + str(c) + '_fitted.png')
    plt.close()

# Visualize Thickness vs Molecular Weight relationship
# Relationship at each concentration is saved to the same figure
def vis_combined_thickness_vs_mw(mws, all_thicknesses, all_err_bars, all_p, all_yhat, all_rsq, concentrations, save_folder):
    # PLOT - LINE
    # Scatter points
    for i in range(len(concentrations)):
        plt.scatter(mws, all_thicknesses[i], color=default_colors[i])
    
    # Line + errorbars
    for i in range(len(concentrations)):
        plt.plot(mws, all_thicknesses[i], color=default_colors[i])
        plt.errorbar(mws, all_thicknesses[i], all_err_bars[i], color=default_colors[i], capsize=7.5)
    
    plt.title('Thickness vs Molecular Weight')
    plt.xlabel('Molecular Weight (amu)')
    plt.ylabel('Thickness (nm)')
    plt.xticks([400000*i for i in range(6)])
    plt.ticklabel_format(style='plain')
    labels = [str(c) + ' (mg/mL)' for c in concentrations]
    plt.legend(labels)
    plt.tight_layout()
    plt.savefig(save_folder + 'combined_t_vs_MW.png')
    plt.close()

    # PLOT - FIT
    # Scatter points
    for i in range(len(concentrations)):
        plt.scatter(mws, all_thicknesses[i], color=default_colors[i])
    
    # Line + errorbars
    for i in range(len(concentrations)):
        plt.plot(mws, all_yhat[i], color=default_colors[i])
        plt.errorbar(mws, all_thicknesses[i], all_err_bars[i], color=default_colors[i], capsize=7.5, linestyle='')
    
    plt.title('Thickness vs Molecular Weight')
    plt.xlabel('Molecular Weight (amu)')
    plt.ylabel('Thickness (nm)')
    plt.xticks([400000*i for i in range(6)])
    plt.ticklabel_format(style='plain')
    labels = [str(concentrations[i]) + ' mg/mL (R' + two_sup + '=' + str(round(all_rsq[i],3)) + ')' for i in range(len(concentrations))]
    plt.legend(labels)
    plt.tight_layout()
    plt.savefig(save_folder + 'combined_t_vs_MW_fitted.png')
    plt.close()

# Visualize Thickness vs Concentration relationship
# Relationship at each mmolecular weight is saved to a new figure
# Contains equations for each relationship
# Contains calculated critical concentration points that are used to train the manifold
def vis_thickness_vs_concentration(mw, concentrations, thicknesses, err_bars, err_bars_cc, ccs, c_thicknesses, p, coeffs, rsq, save_folder):
    # Get points for polynomial-fit curve
    all_x = copy.deepcopy(concentrations)
    all_x.extend(ccs)
    min_x = min(all_x)
    max_x = max(all_x)
    num_points = 25
    fitted_x = np.linspace(min_x, max_x, num_points)
    fitted_y = p(fitted_x)


    plt.scatter(concentrations, thicknesses, label='Experimental Data', color=default_colors[0])
    plt.errorbar(concentrations, thicknesses, err_bars, capsize=7.5, linestyle='', color=default_colors[0])
    plt.plot(fitted_x, fitted_y, label='Polynomial Fit', color=default_colors[1])
    plt.scatter(ccs, c_thicknesses, label='Critical Points', color=default_colors[1])
    plt.errorbar(ccs, c_thicknesses, err_bars_cc, capsize=7.5, linestyle='', color=default_colors[1])

    plt.suptitle(mw + ' - Thickness vs Concentration')
    equation = coeffs_to_quadratic_equation(coeffs)
    plt.title(equation + '    R' + two_sup + '=' + str(round(rsq, 3)))
    plt.xlabel('Concentration (mg/mL)')
    plt.ylabel('Thickness (nm)')

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_folder + 'MW_' + mw + '.png')
    plt.close()

# Visualize Thickness vs Concentration relationship
# Relationship at each mmolecular weight is saved to the same figure
def vis_combined_thickness_vs_concentration(mws, all_concentrations, all_thicknesses, all_err_bars, all_curves, all_rsq, save_folder):
    # Calculate points for curve
    all_x = []
    all_y = []
    for i in range(len(all_concentrations)):
        x = np.linspace(min(all_concentrations[i]), max(all_concentrations[i]), 25)
        y = all_curves[i](x)
        all_x.append(x)
        all_y.append(y)

    # Colors for plotting (so scatter points and curve are matching)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Plot Points
    for i in range(len(all_concentrations)):
        plt.scatter(all_concentrations[i], all_thicknesses[i], color=colors[i], label=mws[i] + ' (R' + two_sup + '=' + str(round(all_rsq[i], 3)) + ')', s=15)
    
    # Plot curves
    for i in range(len(all_x)):
        plt.plot(all_x[i], all_y[i], color=colors[i])
        plt.errorbar(all_concentrations[i], all_thicknesses[i], all_err_bars[i], color=colors[i], capsize=7.5, linestyle='')

    plt.title('Thickness vs Concentration')
    plt.xlabel('Concentration (mg/mL)')
    plt.ylabel('Thickness (nm)')

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_folder + 'combined_t_vs_c.png')
    plt.close()

# Visualize percentage difference in thickness of annealed samples
def vis_annealed_percentage_difference_line(df, save_path, linestyle):
    # Thickness vs concentration graph
    concentrations = [10,15,20,25,30]  # xvals
    mws = ['50K', '311K', '650K']
    df.set_index(keys='MW', inplace=True)

    plt.title('Annealed Samples - Thickness vs Concentration')
    #plt.plot(concentrations, [0 for _ in concentrations], linestyle=linestyle, color='black')
    plt.axhline(0, color='black', linestyle=linestyle, linewidth=0.75)
    plt.xlabel('Concentration (mg/mL)')
    plt.ylabel('Change in Thickness (%)')

    for mw in mws:
        percents = []  # yvals
        for c in concentrations:
            percents.append(df.at[mw, c])
        plt.scatter(concentrations, percents, label=mw)
        plt.plot(concentrations, percents)

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
