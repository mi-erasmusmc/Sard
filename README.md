# Sard

<<<<<<< HEAD
Replication of SARD model from paper: Deep Contextual Clinical Prediction with Reverse Distillation by Kodialam et
al. [1]

### How to Install

    * Install python dependancies from the requirements.txt file
    * Install R requirements from renv.lock file
=======
Replication of SARD model from paper: Deep Contextual Clinical Prediction with Reverse Distillation by Kodialam et al. [1]

### How to Install
    * Install python dependancies from the requirements.txt file with: 
        pip install -r requirements.txt
    * Install R requirements from renv.lock file with
        renv::restore()
>>>>>>> 14e62e9135c625f1210f08955a233cbcfc075d66

### How to Run

* First cohorts need to be defined, either with sql or an OHDSI tool like [ATLAS](http://github.com/OHDSI/Atlas).
* Then they need to be saved in a cohort table with the following columns:
<<<<<<< HEAD
  cohort_definition_id, subject_id, cohort_start_date, cohort_end_date. There will be two cohorts, the target and
  outcome cohort with different id's.
* You need to configure a config.yml file with your database path and access credentials
* Then you can run the R script extractPLPData.R from the command line:  
  `Rscript extractPLPData.R cohortSchema cohortTable cohortId outcomeId pathToOutput pathToPython`
=======
    cohort_definition_id, subject_id, cohort_start_date, cohort_end_date.
  There will be two cohorts, the target and outcome cohort with different id's.
* You need to configure config.yml with your database path and access credentials
* Then you can run the R script extractPLPData.R from the command line:  
`Rscript extractPLPData.R cohortSchema cohortTable cohortId outcomeId pathToOutput pathToPython`
>>>>>>> 14e62e9135c625f1210f08955a233cbcfc075d66
    * cohortSchema is the schema where you have saved your cohorts
    * cohortTable is the table with your cohorts
    * cohortId is the cohort_definition_id of your target cohort
    * outcomeId is the cohort_definition_id of your outcome cohort
<<<<<<< HEAD
    * pathToOutput is the folder where to save output, should be something like './data/task'
    * pathToPython is the path to your python environment
* Then you can run main.py, you need to modify the first lines in the main part to point to the saved data
    * The data needs to be saved in './data/task/' where task is the prediction task
    * You select one of the models in /models to run on the task. Currently support models are:
        - [RETAIN](http://arxiv.org/abs/1608.05745) with some modifications
            - A bidirectional LSTM is used, and the continuous/non-temporal features are concatenated to visit
              embeddings
        - A Transformer from [1]. With addition of non-temporal features being concatenated to embeddings
        - SARD from [1]. With same addition as for the Transformer
        - [Variational GNN](http://arxiv.org/abs/1912.03761) (experimental, hasn't been tested thoroughly yet)

Currently, this version needs the develop branch of FeatureExtraction
from [here](http://github.com/OHDSI/FeatureExtraction/tree/develop).

References

1. Kodialam RS, Boiarsky R, Lim J, Dixit N, Sai A, Sontag D. Deep Contextual Clinical Prediction with Reverse
   Distillation. Proc AAAI Conf Artif Intell 2020;35:249-58 
=======
    * pathToOutput is the folder where to save output
    * pathToPython is the path to your python enviroment
* Then you can run SARD_distill.py, you need to modify the first lines in the main part to point to the saved data
    
Currently this version needs a specific version of FeatureExtraction from [here](http://github.com/egillax/FeatureExtraction).


References

1.  Kodialam RS, Boiarsky R, Lim J, Dixit N, Sai A, Sontag D. Deep Contextual Clinical Prediction 
    with Reverse Distillation. Proc AAAI Conf Artif Intell 2020;35:249-58 
>>>>>>> 14e62e9135c625f1210f08955a233cbcfc075d66

