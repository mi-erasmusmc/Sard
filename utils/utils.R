# A function I use to do my own windowing to compare results with sequence data to normal plp models.
timeWindowing <- function(plpData, windows){

maxTimeBeforeIndex <- max(abs(windows))
  
covariates <- plpData$covariateData$covariates %>% collect()
cohorts <- plpData$cohorts %>% collect()
joined <- covariates %>% inner_join(cohorts %>% select(rowId, cohortStartDate), by='rowId') %>%
  mutate(daysBeforeIndex = cohortStartDate - covariateStartDate) %>%
  filter(((daysBeforeIndex > 0 &  daysBeforeIndex <= maxTimeBeforeIndex) | is.na(daysBeforeIndex)))
timePeriod <- data.table::data.table(relTime= maxTimeBeforeIndex:1, timeId=1:maxTimeBeforeIndex)

newCovariates <- joined %>% mutate(daysBeforeIndex = as.integer(daysBeforeIndex)) %>%
  left_join(timePeriod, by=c("daysBeforeIndex" = "relTime")) %>%
  select(-daysBeforeIndex)

  plpData$covariateData$covariates <- newCovariates
return(plpData)

}

# function I use to remove samples that don't have any temporal data
removeNonTemporalOnly <- function(plpData, population) {

    # remove those that only have age and sex
    covariates <- plpData$covariateData$covariates
    n_covariates_per_row <- covariates %>% group_by(rowId) %>% summarize(unique= n_distinct(covariateId))
    row_Id <- n_covariates_per_row %>% filter(unique>2) %>% select(rowId)

    originalCohortSize <- plpData$cohorts %>% nrow()
    plpData$cohorts <- plpData$cohorts %>% filter(rowId %in% row_Id$rowId)
    plpData$outcomes <- plpData$outcomes %>% filter(rowId %in% row_Id$rowId)

    newCovariates <- covariates %>% filter(rowId %in% !! row_Id$rowId)
    population <- population %>% filter(rowId %in% row_Id$rowId)

    plpData$covariateData$covariates <- newCovariates
    ParallelLogger::logInfo("Removed ", originalCohortSize - nrow(row_Id), " subjects that had only non-temporal features")
    plpData$population <- population
    return(plpData)

}