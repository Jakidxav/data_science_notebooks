# An Image Classifier in Pytorch

This is an application that I build as part of the Udacity Data Science Nanodegree program. It shows how to do transfer learning in PyTorch, where we load a pre-trained network, give it our own classifier, and then save the model for later use.

The Jupyter Notebook showing this process was then transferred over to a command line application which supports user input and is ready for use! To run the application from the shell, first run `train.py` which will save the model, and then run `predict.py` which will output predictions in the terminal. I think that a large part of software development in the future will be incoporating embedded models like the one shown here into smartphone apps or web applications. This project demonstrates the first steps in this process.


To run these files, you will definitely need current versions of these Python libraries:
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- ArgParse
- JSON
- Collections
