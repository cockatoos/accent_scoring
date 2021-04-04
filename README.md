# How to start training
## Getting Azure config file
1. go to https://portal.azure.com/
2. select a Machine Learning resource(If you don't have one, make one!)
3. download config.json file from Overview tab and put it in .azureml folder
![](https://docs.microsoft.com/en-us/azure/machine-learning/media/how-to-configure-environment/configure.png)

## Download data
1. download the dataset from https://www.kaggle.com/mozillaorg/common-voice/data

##Pre-process data and upload to Azure
1. run `python src/generate_datasets.py` the paths can be set by setting the arguments. Check the arguments by running with `-h` flag
##Upload pre-processed data to AzureML's datastore
1. Referencing ./upload_data.py file, you can upload the pre-processed data to your own AzureML linked datastore
2. Make your own dataset through your AzureML studio. Select the dataset tab, and generate dataset from datastore

## Run training algorithm using AzureML
1. run `python train_on_azure.py` you might have to change some parameters in function `Dataset.get_by_name()`, depending on how you named your dataset on previous step

##Result
The trained model would be registered in you AzureML. Find it in Models tab.
# Methods
We are using CNN after converting audio accent files to images using peak extraction.

This data would be used
http://accent.gmu.edu/howto.php  
*kaggle link: https://www.kaggle.com/rtatman/speech-accent-archive

the following repo is used to pre-process the accents audio-file, and maybe distinguish accents
https://github.com/libphy/which_animal
