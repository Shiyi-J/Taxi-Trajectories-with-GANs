# Taxi-Trajectories-with-GANs
### Command inputs:
1.model: CTGAN, RGAN or TimeGAN
2.cate: taxi call type, A, B or C
3.seq_len: length of recorded location points in a trajectory
4.batch_size: number of samples in a batch
5.epochs: iteration of training (for CTGAN epochs refer to the number of generated samples)
6.parameter: dictionary with corresponding hyperparameters
7.plot: whether to plot sample generated trajectories on the map or not

### Model-specific hyperparameters (parameter):
#### CTGAN:
1.it: number of iterations for sampling data (2000)
2.repeat: epochs for training CTGAN (150)
3.n_sample: number of generated samples from a trained model (10)
#### RGAN:
1.test_sz: fraction of samples for testing (0.2)
2.noise_sz: noise dimension for generator input (2)
3.in_sz: input dimension for discriminator (2)
4.out_sz: output dimension for discriminator (2)
5.hidden_sz: hidden unit dimension (100)
6.num_layer: number of layers for lstm (1)
#### TimeGAN:
1.module: gru, lstm or lstmLN (lstmLN)
2.hidden_dim: hidden unit dimension (40)
3.num_layer: number of layers for selected module (2)

### Example commands:
1.`$ python3 --model CTGAN --cate A --seq_len 40 --epochs 10 --parameter "{'it':2000, 'repeat':150, 'n_sample':10}" --plot True`

2.`$ python3 --model RGAN --cate A --seq_len 40 --epochs 500 --batch_size 500 --parameter "{'test_sz':0.2, 'noise_sz':2, 'in_sz':2, 'out_sz':2, 'hidden_sz':100, 'num_layer':1}" --plot True`

3.`$ python3 --model TimeGAN --cate A --seq_len 40 --epochs 5000 --batch_size 128 --parameter "{'module':'lstmLN', 'hidden_dim':40, 'num_layer':2}" --plot True`
