from imageai.Prediction.Custom import ModelTraining

model_trainer = ModelTraining() # type of training algorithm in this case basic NN
model_trainer.setModelTypeAsResNet() # defines the type of model that will be stored, in this case it will be a simple .h5 file
model_trainer.setDataDirectory("idenprof") # where the AI will look for data to train it
model_trainer.trainModel(num_objects=10, num_experiments=200, batch_size=32, show_network_summary=True)
# num_objects is the total number of different types of objects, eg: chef, car, cat
# num_experiments is the total number of epochs or the total number of times an "experiment" is run
# batch_size is the total number of images tested in an epoch
