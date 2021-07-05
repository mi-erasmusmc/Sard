timeWindowing <- function(plpData, windows){


maxTimeBeforeIndex <- max(abs(windows))
  
covariates <- plpData$covariateData$covariates %>% collect()
cohorts <- plpData$cohorts %>% collect()
joined <- covariates %>% inner_join(cohorts %>% select(rowId, cohortStartDate), by='rowId') %>%
  mutate(daysBeforeIndex = cohortStartDate - as.Date(covariateStartDate)) %>%
  filter(((daysBeforeIndex > 0 &  daysBeforeIndex <= maxTimeBeforeIndex) | is.na(daysBeforeIndex)))
timePeriod <- data.table::data.table(relTime= maxTimeBeforeIndex:1, timeId=1:maxTimeBeforeIndex)

newCovariates <- joined %>% mutate(daysBeforeIndex = as.integer(daysBeforeIndex)) %>%
  left_join(timePeriod, by=c("daysBeforeIndex" = "relTime")) %>%
  select(-daysBeforeIndex)

  plpData$covariateData$covariates <- newCovariates
return(plpData)

}

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
}