# accent_scoring
Accent scoring using Azure AI


## Getting Azure config file
1. go to https://portal.azure.com/
2. select Cockatoos Machine Learning resource
3. download config.json file from Overview tab
![](https://docs.microsoft.com/en-us/azure/machine-learning/media/how-to-configure-environment/configure.png)

## Downloading data
visit https://www.kaggle.com/rtatman/speech-accent-archive, and download.

## Methods
We are using CNN after converting audio accent files to images using peak extraction.

This data would be used
http://accent.gmu.edu/howto.php
*kaggle link: https://www.kaggle.com/rtatman/speech-accent-archive

the following repo is used to pre-process the accents audio-file, and maybe distinguish accents
https://github.com/libphy/which_animal
