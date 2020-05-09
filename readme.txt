This is the program of final year project of Generative Adversarial Networks for Defensing Adversarial Examples.
The reproduced Defense-GAN and Tmp-Defense-GAN be implemented in this program.
Referenced the Defense-GAN from https://github.com/kabkabm/defensegan

Student: Maizhen Ning
ID: 201376369 

Requirements: Numpy, TensorFlow2.0, matplotlib

If you want to test the program, white_box and black_box are provided for testing algorithm. 
You can directly run the file to see result. (change the direction of terminal to this folder please)

If you want to change hyperparameters, please open the file and see the last part of code, an instruction be provided for you to set test conditions. 
Related hyperparameters can be set as the name of parameters already be sppecified in program.
If the required Defense-GAN model existed, the algorithm will restroe the checkpoint. If not exist, algorithm will start train a new one. 
(There is just a pre-trainend FMNIST Defense-GAN be provided, because large file cannot be submitted online, MNIST defense-GAN need be trained by yourself)

If you meet problems, please contact me: m.ning@student.liverpool.ac.uk

COMP39X Final Year Project
Department of Computer Science
University of Liverpool
