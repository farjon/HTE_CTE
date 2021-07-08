# HTE experiments - hyperparameters setting

## these experiments were meant to check which architecture and settings gives the best results

###follow these intrucations to perform an experiment
0) use the HTE_line_serach.py file to run the experiment
1) choose a parameter that you would like to examine
   - NOF : number of ferns (might have different number of ferns is each layer)
   - NOBF: number of bit functions (might have different number of ferns is each layer's ferns)
   - NOL : number of layers
   - NODO: number of output features from the tables (might have different D_out is each layer)
2) choose the values for the experimented parameter

3) choose an experiment number (notice that if the eperiment number already exist, it will override existing folders)
4) choose the dataset on which the experiment will be conducted
5) choose the architecture to use:
    - there are several architectures that we would like to examine, these are:
        - batch_norm  : including batch normalization between layers
        - ResFern     : using skip connections between layers
        - ResFern_conn: concatenating the input features into each input vector before feed-forwarding it
        - regression  : solving a regression problem, here we might use any of the above and also consider:
            - Linear_1 : using a single output neuron and using the MAE or MSE loss functions
            - Linear_2 : using 2 output neuron and using the uncertinty loss function

The experiment results and description will be saved in the storagePath
define you storage path at GetEnvVar.py

