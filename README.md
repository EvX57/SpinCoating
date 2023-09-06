# SpinCoating

## This repository contains the code supplement for *Utilizing Machine Learning to Model Interdependency of Bulk Molecular Weight, Solute Concentration, and Thickness of Spin Coated Polystyrene Thin Films*.

`Data` contains all the raw experimental data and processed data.

`preprocess.py` can be used to process the raw experimental data for visualization or training of the machine learning model.

`visualize.py` can be used to visualize the data.

`model.py` contains the machine learning model to predict critical concentration from thickness and molecular weight. `model_train()` trains the model, `model_evaluate()` evaluates the model's performance on the test datapoints we collected, and `model_predict()` predicts the critical concentration for new thickness and molecular weight values.
