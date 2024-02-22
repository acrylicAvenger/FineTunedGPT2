This project was an attempt to make ChatGPT like Q$A bot as a part of my college project.

The project consists of 4 files.

1. main.py consists of the code to build and save the model.
2. apiGPT2.py consists of code (using Django) that hosts the saved model on localhost port number 6500.
3. dummydataset.txt is a sample dataset text file to train the model
4. "retrain the model.py" is to retrain the saved model for better fine tuning.
5. Index.js is for the REACT app to display the ouput.

Note: please update the modelpath, tokenizerpath and dataset location according to your file locations.


#How to use:
1. Install all the dependencies.
2. Run main.py file after updating the dataset directories.
3. Run apiGPT3.py after updating the savedmodel directory location.
4. Copy the url of the server and paste it in the REACT api code
5. Build and run the REACT app
   
