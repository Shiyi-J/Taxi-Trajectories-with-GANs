# Synthesizing Taxi Trajectories using GANs
This project will generate taxi trajectories using three different GANs and will provide dynamic time warping and maximum mean discrepancy scores for evaluation. Plots of sample synthetic routes will be provided if choosing `--plot True`.

### Note:
Dataset used for this project is from Kaggle ECML/PKDD 15: Taxi Trajectory Prediction (I) problem. We used train.csv.zip file in this project. The link for downloading the dataset is provided below:
https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data

Before running the command, you need to change the path `D = pd.read_csv('~/work/Taxi_Trajectories/train_taxi_trajectory.csv')` in the main function in Main.py to where you save downloaded dataset locally. Please check Requirements.txt as well to see if all the requirements are satisfied for running the program.

### Command inputs:
-model: CTGAN, RGAN or TimeGAN

-cate: taxi call type, A, B or C

-seq_len: length of recorded location points in a trajectory

-batch_size: number of samples in a batch

-epochs: iteration of training (for CTGAN epochs refer to the number of generated samples)

-parameter: dictionary with corresponding hyperparameters

-plot: whether to plot sample generated trajectories on the map or not

### Model-specific hyperparameters (parameter):
#### CTGAN:
-it: number of iterations for sampling data (2000)

-repeat: epochs for training CTGAN (150)

-n_sample: number of generated samples from a trained model (10)

#### RGAN:
-test_sz: fraction of samples for testing (0.2)

-noise_sz: noise dimension for generator input (2)

-in_sz: input dimension for discriminator (2)

-out_sz: output dimension for discriminator (2)

-hidden_sz: hidden unit dimension (100)

-num_layer: number of layers for lstm (1)

#### TimeGAN:
-module: gru, lstm or lstmLN (lstmLN)

-hidden_dim: hidden unit dimension (40)

-num_layer: number of layers for selected module (2)

### Example commands:
1.`$ python3 Main.py --model CTGAN --cate A --seq_len 40 --epochs 10 --parameter "{'it':2000, 'repeat':150, 'n_sample':10}" --plot True`

2.`$ python3 Main.py --model RGAN --cate A --seq_len 40 --epochs 500 --batch_size 500 --parameter "{'test_sz':0.2, 'noise_sz':2, 'in_sz':2, 'out_sz':2, 'hidden_sz':100, 'num_layer':1}" --plot True`

3.`$ python3 Main.py --model TimeGAN --cate A --seq_len 40 --epochs 5000 --batch_size 128 --parameter "{'module':'lstmLN', 'hidden_dim':40, 'num_layer':2}" --plot True`
