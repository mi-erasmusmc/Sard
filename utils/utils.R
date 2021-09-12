<<<<<<< HEAD
# A function I use to do my own windowing to compare results with sequence data to normal plp models.
=======
>>>>>>> 14e62e9135c625f1210f08955a233cbcfc075d66
timeWindowing <- function(plpData, windows){


maxTimeBeforeIndex <- max(abs(windows))
  
covariates <- plpData$covariateData$covariates %>% collect()
cohorts <- plpData$cohorts %>% collect()
joined <- covariates %>% inner_join(cohorts %>% select(rowId, cohortStartDate), by='rowId') %>%
<<<<<<< HEAD
  mutate(daysBeforeIndex = cohortStartDate - covariateStartDate) %>%
=======
  mutate(daysBeforeIndex = cohortStartDate - as.Date(covariateStartDate)) %>%
>>>>>>> 14e62e9135c625f1210f08955a233cbcfc075d66
  filter(((daysBeforeIndex > 0 &  daysBeforeIndex <= maxTimeBeforeIndex) | is.na(daysBeforeIndex)))
timePeriod <- data.table::data.table(relTime= maxTimeBeforeIndex:1, timeId=1:maxTimeBeforeIndex)

newCovariates <- joined %>% mutate(daysBeforeIndex = as.integer(daysBeforeIndex)) %>%
  left_join(timePeriod, by=c("daysBeforeIndex" = "relTime")) %>%
  select(-daysBeforeIndex)

  plpData$covariateData$covariates <- newCovariates
return(plpData)

}

<<<<<<< HEAD
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
=======
removeNonTemporalOnly <- function(plpData, outcomeId) {
# remove those that only have nonTemporal covariates
covariates_per_row <- plpData$covariateData$covariates %>% group_by(rowId) %>% summarize(unique= n_distinct(covariateId),
                                                                     naTimeId = sum(is.na(timeId)))
includedRowIds <-covariates_per_row %>% filter(unique > naTimeId) %>% select(rowId) %>% collect()

originalCohortSize <- plpData$cohorts %>% nrow()
plpData$cohorts <- plpData$cohorts %>% filter(rowId %in% includedRowIds$rowId)
plpData$outcomes <- plpData$outcomes %>% filter(rowId %in% includedRowIds$rowId)

included <- includedRowIds$rowId
plpData$covariateData$covariates <-  plpData$covariateData$covariates %>% filter(rowId %in% included) %>% collect()
ParallelLogger::logInfo("Removed ", originalCohortSize - nrow(includedRowIds), " subjects that had only non-temporal features")

# add attrition metadata
metaData <- attr(plpData$cohorts, 'metaData')

attrRow <- plpData$cohorts %>% dplyr::group_by() %>%
  dplyr::summarise(outcomeId = outcomeId,
                   description = 'After removal of subjects with only non-temporal covariates',
                   targetCount = length(.data$rowId),
                   uniquePeople = length(unique(.data$subjectId)),
                   outcomes = sum(!is.na(.data$cohortId == outcomeId), 
                                  na.rm = TRUE)) 
metaData$attrition <- rbind(metaData$attrition, attrRow)

attr(plpData$cohorts, 'metaData')$attrition <- metaData$attrition

return(plpData)
>>>>>>> 14e62e9135c625f1210f08955a233cbcfc075d66
}