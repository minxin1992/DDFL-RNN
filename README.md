# DDFL-RNN
DDFL-RNN Resource files

Due to privacy and other reasons, the data cannot be fully uploaded, we have uploaded sample data and full code.
## Environment Settings
We use pytorch to implement our method. 
- Torch version:  '1.0.1'
- Python version: '3.8'
## A quick start to run the codes:

Train:

'DDPL-RNN.py', './data/sample_train_cp_data.csv', './data/sample_test_cp_data.csv', 'CP_diag_rec', 1, 1

The above command will train our model based on 100 patients' sample data

Test:

'DDPL-RNN.py', './data/sample_train_cp_data.csv', './data/sample_test_cp_data.csv', 'CP_diag_rec', 1, 0

We just need to change the mode flag from 1 to 0. The performance of the model on the validation set will be printed out.

The code and sample data ensure the authenticity and reproducibility of the method.

If you required the full dataset and have any other questions, please feel free to communicate with the corresponding author or the first author.
