# Machine Learning Python Files
Various versions of the ML algorithms coorespond to the different stages of the development of this project.
</br></br>
## Stage 1
FormatConversion.ipynb, ML_Exploratory.ipynb and First Recordings.ipynb are first created to test the XDawn model on the BCI Competition dataset as well as the first self recordings.
- FormatConversion.ipynb was used to convert the original format of the BCIC dataset to one similar to what we used in the actual app. 
- ML_Exploratory.ipynb tests the Xdown model on the reformatted BCIC dataset
- First Recordings tests the same ML model on the first self recorded dataset
- generate_epoch.py was the helper function used for generating the epochs during this stage

## Stage 2
During this stage we formalized the format of the modeling algoritms and transitioned into Powerbin classification instead of XDawn. Data visualization files were developed to help understanding the model effectiveness.
- GenerateMLModel.ipynb was the first reformmated modeling file that allows multiple models to follow the same format
- GenerateMLMedelBCIC.ipynb tested the performance of it on the BCIC dataset
- DataVisualization.ipynb and DataVisualization_BCIC.ipynb explores the PSD feature of self recorded dataset as well as the BCIC dataset

## Stage 3
This stage focuses on the data visualization as well as implementing the final Powerbin model used in the final app.
- PowerBin_Vis_Model_Self-Recorded.ipynb is developed to extensively verify the validity of the powerbin ratio model
- Self-Recorded_Modeling.ipynb is the final modeling software used in the app
- helper functions for eeg signal processing are integrated into a separate file called data_processing_helper.py
