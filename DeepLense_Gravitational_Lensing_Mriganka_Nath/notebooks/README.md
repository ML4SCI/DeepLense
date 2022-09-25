## Tutorial Notebooks
* These notebooks show you how to use the differnt domain adaptation algorithms. I have only shown notebook code for the best encoder
for each domain adaptation technique. 
* Changing to other encoders is very easy, just change the name in the Encoder class. Look 
at model.py for more details


## Files

|Notebooks                  | Description                         |
|------------------------|-------------------------------------|
|adamatch-effnet         | AdaMatch algorithm using efficientnet_b2 encoder     |
|adda-ecnn               | ADDA algorithm using Equivariant Convolutional Neural Network encoder     |
|self-ensembling-effnet  | Self-ensembling algorithm using efficientnet_b2 encoder     |
|lensing-data-xploration | Exploratory data analysis of the observational/real data and insights|

### Note
While the simulated data directly downloads as shown in the notebooks, but for the Real data you have to
first run the lensing-data-xploration notebook it will output a csv file, we will use this csv in the notebooks. 
             