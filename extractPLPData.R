library(PatientLevelPrediction)
source('utils/utils.R')
library(config)

#' Extract features from cohorts and save for modelling in pytorch
#' 
#' @param cohortSchema    The schema where the cohort table is
#' @param cohortTable     Table with Cohorts
#' @param cohortId        The cohort_definition_id of target cohort
#' @param outcomeId       The cohort_definition_id of outcome cohort
#' @param pathToOutput    Where to save output
#' @param pathToPython    Path to python binary for reticulate to use. Needs to 
#'                        have pytorch installed.
#' 
extractPLPData <- function(cohortSchema='mySchema', cohortTable='myTable', 
                           cohortId=4, outcomeId=9999,
                           pathToOutput='./data',
                           pathToPython='./venv/bin/python'){

config <- get('database')

connectionDetails <- createConnectionDetails(dbms=config$dbms,
                                              server=config$path,
                                              user=config$user,
                                              password=config$password,
                                              pathToDriver=config$driver)

<<<<<<< HEAD
windows <- c((-3*365):1) # extract last three years, 1 window per day
tempCovarSettings <- createTemporalSequenceCovariateSettings(useDemographicsGender= TRUE,
                                                             useDemographicsAge=TRUE,
                                                             useConditionOccurrence=TRUE,
                                                             useDrugExposure=TRUE,
                                                             useProcedureOccurrence=TRUE,
                                                             timePart = 'DAY',
                                                             timeInterval = 1,
                                                             sequenceStartDay = -365,
                                                             sequenceEndDay = -1
=======
windows <- c((-3*365):-11) # extract last three years, 1 window per day
tempCovarSettings <- createTemporalCovariateSettings(useDemographicsGender= TRUE,
                                                     useDemographicsAge=TRUE,
                                                     useConditionEraStart=TRUE,
                                                     useDrugExposure=TRUE,
                                                     useProcedureOccurrence=TRUE,
                                                     temporalStartDays=windows,
                                                     temporalEndDays=windows
                                                     )
>>>>>>> 14e62e9135c625f1210f08955a233cbcfc075d66

# get plp data with my already defined cohorts in my cohortTable
plpData <- getPlpData(connectionDetails = connectionDetails,
                      cdmDatabaseSchema = 'cdm',
                      cohortDatabaseSchema = cohortSchema,
                      cohortTable = cohortTable,
                      cohortId = cohortId,
                      outcomeIds = outcomeId,
                      covariateSettings = tempCovarSettings,
                      outcomeDatabaseSchema = cohortSchema,
                      outcomeTable = cohortTable,
<<<<<<< HEAD
                      firstExposureOnly = FALSE)
=======
                      firstExposureOnly = TRUE)
>>>>>>> 14e62e9135c625f1210f08955a233cbcfc075d66


#sort features in windows 
plpData <- timeWindowing(plpData, windows = windows)

<<<<<<< HEAD
# non-temporal features have timeId of NA which gives issues for some joins, so I replace with -1timeId
plpData$covariateData$covariates <- tidyr::replace_na(plpData$covariateData$covariates, list(timeId=-1))

# normalize continous features and remove rare features
plpData$covariateData  <- FeatureExtraction::tidyCovariateData(plpData$covariateData)

=======
# normalize continuous features and remove rare features
plpData$covariateData  <- FeatureExtraction::tidyCovariateData(plpData$covariateData)

# remove those that only have demographics, removed from covariates, cohorts and outcome tables
plpData <- removeNonTemporalOnly(plpData, outcomeId)

# save plpData object
outputFolder <- './data/plp_output/'
dir.create(outputFolder, recursive = TRUE)

savePlpData(plpData, outputFolder)

plpData <-  loadPlpData('./data/plp_output/')

>>>>>>> 14e62e9135c625f1210f08955a233cbcfc075d66
# create study population
population <- createStudyPopulation(plpData=plpData,
                                    outcomeId = outcomeId,
                                    firstExposureOnly = TRUE,
                                    requireTimeAtRisk = TRUE,
                                    minTimeAtRisk = 1,
                                    riskWindowEnd = 365,
<<<<<<< HEAD
                                    includeAllOutcomes = TRUE,
                                    removeSubjectsWithPriorOutcome = TRUE,
                                    verbosity = 'DEBUG')

# remove those that only have demographics, removed from covariates, cohorts and outcome tables
plpData <- removeNonTemporalOnly(plpData,population)
population <- plpData$population

# save plpData object
outputFolder <- pathToOutput
dir.create(outputFolder, recursive = TRUE)

savePlpData(plpData, outputFolder)

# restrict data to population and re-index to minimize size of sparse matrix
population$rowIdPython <- population$rowId-1
indices <- population %>% select(rowId, rowIdPython)
plpData$covariateData$covariates <- plpData$covariateData$covariates %>% inner_join(indices, by='rowId', copy=TRUE)

# do test-val-train split and add to population dataframe so all analyses whether in R or python use same split
# if each sample is a visit, use subjectSplitter so same patient is only in either train or test set but not both
# if each sample is a patient, use randomSplitter for stratitifed split
# indices <- subjectSplitter(population, test=0.25, nfold=3, seed=42)
indices <- randomSplitter(population, test=0.25, nfold=3, seed=42)
population <- population %>% inner_join(indices)

# use python to create a sparse matrix compatible with pytorch
reticulate::use_python(pathToPython)
x_torch <- toSparseTorchPython(plpData, population, map=NULL, temporal=TRUE,
                               nonTemporalCovs=FALSE)
x_torch$population <- population

# save in pytorch format
torch <- reticulate::import('torch')
torch$save(x_torch, file.path(outputFolder, 'python_data'))
=======
                                    inlcludeAllOutcomes = TRUE,
                                    removeSubjectsWithPriorOutcome = TRUE,
                                    verbosity = 'DEBUG')


# create new rowIds in population, covariates and cohorts
# so sparse matrix generation doesn't create larger than neccesary matrix
mappingTable <- data.frame(rowId=population$rowId, newRowId=1:nrow(population))
population <- population %>% inner_join(mappingTable, by="rowId") %>% select(!rowId) %>%
  rename(rowId=newRowId)
plpData$cohorts <- plpData$cohorts %>% inner_join(mappingTable, by="rowId") %>% select(!rowId) %>%
  rename(rowId=newRowId)
plpData$covariateData$covariates <- plpData$covariateData$covariates %>% 
  collect() %>% left_join(mappingTable, by="rowId") %>% select(!rowId) %>%
  rename(rowId=newRowId)



# use python to create a sparse matrix compatible with pytorch
population$rowIdPython <- population$rowId-1
reticulate::use_python('~/PycharmProjects/test_project/venv/bin/python')
startTime <- Sys.time()
x_torch <- toSparseTorchPython(plpData, population, map=NULL, temporal=T,
                               nonTemporalCovs=F)
endTime <- Sys.time()
endTime - startTime

x_torch$population <- population

# the conversion to python doesn't work with na values in date columns
x_torch$covariates$covariateStartDate <- as.character.Date(x_torch$covariates$covariateStartDate)
x_torch$covariates$covariateEndDate <- as.character.Date(x_torch$covariates$covariateEndDate)
x_torch$covariates <- tidyr::replace_na(x_torch$covariates, list(covariateStartDate=NaN, covariateEndDate=NaN,
                                           timeId=NaN))

# save in pytorch format
torch <- reticulate::import('torch')
torch$save(x_torch, file.path(outputFolder, 'plp_output'))
>>>>>>> 14e62e9135c625f1210f08955a233cbcfc075d66

}

args <- commandArgs(trailingOnly = TRUE)

schema <- args[1]
table <- args[2]
cohortId <- args[3]
outcomeId <- args[4]
outputPath <- args[5]
pythonPath <- args[6]

extractPLPData(cohortSchema=schema, cohortTable=table, cohortId=cohortId,
               outcomeId=outcomeId, pathToOutput = outputPath,
               pathToPython = pythonPath)