import pandas as pd
import numpy as np
from scipy.optimize import minimize
import functools
import math
import matplotlib.pyplot as plt
import preprocess
import preprocess
import statistics
import math

# Symbols for printing
ang_symbol = '\u212B'

# Datapoints are represented as (Thickness, MW, CC)
# Training datapoints for the manifold
def create_datapoints(df, mw_names, mws):
    datapoints = []
    for i, mw in enumerate(mw_names):
        for j in range(len(df)):
            t = df.at[j, 'Thickness']
            cc = df.at[j, mw]
            datapoints.append([t, mws[i], cc])
    return datapoints

# Datapoints are represented as (Thickness, MW, CC)
# Testing datapoints for the manifold
def create_test_datapoints(df, save=True):
    mws = list(df.columns)
    mws.remove('mW')

    datapoints = []
    for i, mw in enumerate(mws):
        c = None
        thicknesses = []
        for j in range(len(df)):
            row = df.at[j, 'mW']
            if row == 'Concentration':
                c = float(df.at[j, mw])
            elif 'Thickness' in row:
                thickness = float(df.at[j,mw])
                if not math.isnan(thickness):
                    thicknesses.append(thickness)
        datapoints.append([statistics.mean(thicknesses), float(mw), c])

    # Save to df
    if save:
        save_df = pd.DataFrame(columns=['Thickness', 'MW', 'Concentration'])
        for i, dp in enumerate(datapoints):
            save_df.loc[i] = dp
        save_df.to_csv('Data/test_data_formatted.csv', index=False)
        return datapoints, save_df
    else:
        return datapoints

def create_datapoints_from_raw(df, mw_names, mws):
    datapoints = []
    for i, mw in enumerate(mw_names):
        for j in range(len(df)):
            cc = df.at[j, 'Concentration']
            t = df.at[j, mw]
            datapoints.append([t, mws[i], cc])
    return datapoints

# Reshapes points for visualization
def reshape(list):
    new_list = []
    for j in range(len(list[0])):
        cur = []
        for i in range(len(list)):
            cur.append(list[i][j])
        new_list.append(cur)
    return new_list

def equation_0(vars, params):
    return (params[0] + vars[0] * params[1] + params[2] * vars[0] ** 2) * params[3] * math.pow(vars[1], params[4])

# Linear relationship between thickness and concentration
# Exponential relationship between molecular weight and concentration
def equation_1(vars, params):
    return (params[0] + vars[0] * params[1]) * params[2] * math.pow(vars[1], params[3])

# Derived relationship for MW in terms of thickness and concentration
# vars[0] - thickness
# vars[1] - concentration
def equation_2(vars, params):
    return math.pow(vars[1] / ((params[0] + vars[0] * params[1]) * params[2]), 1/params[3])

# Derived relationship for thickness in terms of MW and concentration
# vars[0] - concentration
# vars[1] - MW
def equation_3(vars, params):
    return ((vars[0] / (params[2] * math.pow(vars[1], params[3]))) - params[0]) / params[1]

# Error function for fitting
# Mean-squared error
def error(params, points, eq):
    sum = 0
    for p in points:
        input_vals = p[:-1]
        output_val = p[-1]
        sum += (eq(input_vals, params) - output_val) ** 2
    return (sum / len(points))

# Percentage error
def percent_error(act, pred):
    err = []
    for i in range(len(act)):
        error = abs(pred[i] - act[i]) / act[i] * 100
        err.append(error)
    return err

# Trains the model
# Input: thickness and molecular weight
# Output: concentration
# Returns learned coefficients for equation/model
def model_train(df, eq, viz_model=False, save_path=''):
    mw_names = ['30K','50K','123K','200K','311K','650K','1080K','2000K']
    mws = [30000, 50000, 123000, 200000, 311000, 650000, 1080000, 2000000]
    datapoints = create_datapoints_from_raw(df, mw_names, mws)


    # Fit equation
    bounds_1 = [(0,10), (0.0001,0.1), (-5, 5), (10,2000), (-1,0)]
    params0_1 = (5, 0.001, 0, 250, -0.2)
    err_function = functools.partial(error, points=datapoints, eq=eq)
    res = minimize(err_function, params0_1, bounds=bounds_1)
    params = res.x

    # Calculate training error
    y_true = []
    y_pred = []
    for dp in datapoints:
        vars = [dp[0], dp[1]]
        y_true.append(dp[2])
        y_pred.append(eq(vars, params))

    save_df = pd.DataFrame()
    save_df['Concentration'] = y_true
    save_df['Predicted'] = y_pred
    save_df['Percent Error'] = percent_error(y_true, y_pred)
    save_df.to_csv('Data/train_output.csv')
    
    print('\nModel Training Results:')
    print('RMSE: ' + str(preprocess.rmse(y_true, y_pred)))
    print('RMSPE: ' + str(preprocess.rmspe(y_true, y_pred)))
    print('MAPE: ' + str(preprocess.mape(y_true, y_pred)))

    # Visualize
    if viz_model:
        datapoints = reshape(datapoints)
        t_vals = datapoints[0]
        mw_vals = datapoints[1]
        t_grid = np.linspace(min(t_vals), max(t_vals), 25)
        mw_grid = np.linspace(min(mw_vals), max(mw_vals), 25)
        x_mesh, y_mesh = np.meshgrid(t_grid, mw_grid)
        x_flat = np.ravel(x_mesh)
        y_flat = np.ravel(y_mesh)
        cc_vals = np.array([eq([x_flat[i],y_flat[i]], params) for i in range(len(x_flat))])
        cc_vals = cc_vals.reshape(x_mesh.shape)

        ax = plt.axes(projection='3d')
        ax.scatter3D(t_vals, mw_vals, datapoints[2])
        ax.plot_surface(x_mesh, y_mesh, cc_vals, alpha=0.75)

        #ax.view_init(20, 60)  # Viewing angle for saved graph
        ax.view_init(20, 30)

        plt.title('3D Manifold')
        plt.ticklabel_format(style='plain')
        ax.set_xlabel('Thickness (nm)')
        ax.invert_xaxis()
        ax.set_ylabel('Molecular Weight (amu)')
        ax.set_zlabel('Concentration (mg/mL)')
        #ax.set_zlabel('Critical Concentration (mg/mL)')
        plt.show()
        #plt.savefig(save_path)
        plt.close()

    return params

# Evaluate the model on test data
def model_evaluate(df_train, equation, df_test):
    datapoints, save_df = create_test_datapoints(df_test, save=True)
    params = model_train(df_train, equation)

    # Concentration Predictions
    y_true = []
    y_pred = []
    for dp in datapoints:
        y_true.append(dp[2])
        y_pred.append(equation(dp[:-1], params))

    save_df['Predictions'] = y_pred
    save_df['Percent Error'] = percent_error(y_true, y_pred)
    save_df.to_csv('Data/output.csv', index=False) # Put the path where you would like to save the results here
    
    print('\nModel Evaluation Results:')
    print('RMSE: ' + str(preprocess.rmse(y_true, y_pred)))
    print('RMSPE: ' + str(preprocess.rmspe(y_true, y_pred)))
    print('MAPE: ' + str(preprocess.mape(y_true, y_pred)) + '\n')
    
# Predicts concentration for input thicknesses and molecular weights
# Inputs: list of [thickness, mw] datapoints
def model_predict(df_train, equation, inputs, viz_model=False, save_path=''):
    mw_names = ['30K','50K','123K','200K','311K','650K','1080K','2000K']
    mws = [30000, 50000, 123000, 200000, 311000, 650000, 1080000, 2000000]

    params = model_train(df_train, mw_names, mws, equation, viz_model, save_path)

    outputs = [equation(input, params) for input in inputs]
    return outputs

if __name__ == '__main__':
    df_raw = pd.read_csv('Data/averaged.csv')
    df_test = pd.read_csv('Data/test_data_presentation.csv') # Put the path to your data here

    model_evaluate(df_raw, equation_0, df_test)