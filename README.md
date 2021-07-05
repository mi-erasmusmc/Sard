# Sard

Replication of SARD model from paper: Deep Contextual Clinical Prediction with Reverse Distillation by Kodialam et al. [1]

### How to Install
    * Install python dependancies from the requirements.txt file with: 
        pip install -r requirements.txt
    * Install R requirements from renv.lock file with
        renv::restore()

### How to Run

* First cohorts need to be defined, either with sql or an OHDSI tool like [ATLAS](http://github.com/OHDSI/Atlas).
* Then they need to be saved in a cohort table with the following columns:
    cohort_definition_id, subject_id, cohort_start_date, cohort_end_date.
  There will be two cohorts, the target and outcome cohort with different id's.
* You need to configure config.yml with your database path and access credentials
* Then you can run the R script extractPLPData.R from the command line:  
`Rscript extractPLPData.R cohortSchema cohortTable cohortId outcomeId pathToOutput pathToPython`
    * cohortSchema is the schema where you have saved your cohorts
    * cohortTable is the table with your cohorts
    * cohortId is the cohort_definition_id of your target cohort
    * outcomeId is the cohort_definition_id of your outcome cohort
    * pathToOutput is the folder where to save output
    * pathToPython is the path to your python enviroment
* Then you can run SARD_distill.py, you need to modify the first lines in the main part to point to the saved data
    
Currently this version needs a specific version of FeatureExtraction from [here](http://github.com/egillax/FeatureExtraction).


References

1.  Kodialam RS, Boiarsky R, Lim J, Dixit N, Sai A, Sontag D. Deep Contextual Clinical Prediction 
    with Reverse Distillation. Proc AAAI Conf Artif Intell 2020;35:249-58 

