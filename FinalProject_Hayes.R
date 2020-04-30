library(mlr3)
library(sciteR)
library(readxl)
library(tidyverse)
library(mlr3viz)
library(mlr3filters)
library(mlr3learners)
library(mlr3tuning)
library(paradox)
library(GGally)
library(data.table)
library(ggthemes)
library(stargazer)
library(glmnet)
library(e1071)
library(kknn)

# Read data with original classifications
data <- read_xlsx("final_journals_sji_25March.xlsx", sheet = "scite_data") %>%
  arrange(issn) %>%
  filter(!duplicated(issn)) %>%
  select(-c(issn, journal_title))%>%
  mutate(subject = as.factor(subject)) %>%
  filter(subject != "other")
view(data)

# Read data with new consolidated classifications
data_reclassified <- read_xlsx("final_journals_sji_reclassified.xlsx", sheet = "scite_data") %>%
  arrange(issn) %>%
  filter(!duplicated(issn)) %>%
  select(-c(issn, journal_title)) %>%
  mutate(subject = as.factor(subject)) %>%
  filter(subject != "other")
view(data_reclassified)

# Check some info on medicine journals
medicine_data = data_reclassified %>%
  filter(subject == "medicine") %>%
  select(sji_all_years, total_cites)

# See the correlation between sji and total citations
cor(data_reclassified$sji_all_years, data_reclassified$total_cites)

# Re-add the journal name to the data
data_w_journal <- read_xlsx("final_journals_sji_reclassified.xlsx") %>%
  arrange(desc(total_cites)) %>%
  filter(!duplicated(issn)) %>%
  select(-issn) %>%
  mutate(subject = as.factor(subject)) %>%
  filter(subject != "other") %>%
  mutate(journal_title = as.factor(journal_title))
view(data_w_journal)

# Which journals are the most cited?
top_cited_journals <- data_w_journal[1:(0.10*dim(data_w_journal)[1]),1:7]
view(top_cited_journals)
mean(top_cited_journals$sji_all_years)
mean(data_w_journal$sji_all_years)

# change to desc as necessary
journals_arrange_by_sji <- data_w_journal %>%
  arrange(desc(sji_all_years))
highest_sji_journals <- journals_arrange_by_sji[1:(0.10*dim(journals_arrange_by_sji)[1]),1:7]
view(highest_sji_journals)
highest_sji_journals %>%
  filter(subject == "physics")

data_w_journal %>%
  ggplot() +
  geom_point(aes(x = total_cites, y = sji_all_years, )) +
  coord_cartesian(xlim = c(0, 250000)) +
  labs(title = "No Correlation Between Total Cites and SJI")

fraction_contradicting <- data_w_journal$total_contradicting_cites/data_w_journal$total_cites
fraction_supporting <- data_w_journal$total_supporting_cites/data_w_journal$total_cites
plot(fraction_contradicting, fraction_supporting)

cor(fraction_contradicting, data_w_journal$sji_all_years)
cor(fraction_supporting, data_w_journal$sji_all_years)
cor(fraction_contradicting, fraction_supporting)


# Make ML model -------------
task_scite <- TaskClassif$new(id = "scite", backend = data_reclassified, target = "subject")
print(task_scite)

# SJI densities
autoplot(task_scite, type = "pairs")[2,2] +
  scale_fill_economist()


learner <- lrn("classif.rpart")
mlr_learners$get("classif.kknn")
learner_kknn <- lrn("classif.kknn")

train_set = sample(task_scite$nrow, 0.80 * task_scite$nrow)
test_set = setdiff(seq_len(task_scite$nrow), train_set)

learner$train(task_scite, row_ids = train_set)

prediction = learner$predict(task_scite, row_ids = test_set)
prediction$confusion
length(prediction$truth)

# Using Naive Bayes -------------

learner_bayes = lrn("classif.naive_bayes")
learner_bayes$train(task_scite, row_ids = train_set)
prediction_bayes = learner_bayes$predict(task_scite, row_ids = test_set)
prediction_bayes$confusion


# Using KKNN -----------
# 
tuner = tnr("random_search")
measure = lapply(c("classif.ce", "classif.acc"), msr)
resample = rsmp("cv", folds = 6L)
resample$instantiate(task_scite)
evals20 = term("evals", n_evals = 40)

tune_kknn = ParamSet$new(list(
  ParamInt$new("k", lower = 1, upper = 40)
))

instance_kknn = TuningInstance$new(
  task = task_scite,
  learner = learner_kknn,
  resampling = resample,
  measures = measure,
  param_set = tune_kknn,
  terminator = evals20
)

tuner$tune(instance_kknn)
instance_kknn$result

learner_kknn$param_set$values = instance_kknn$result$params

learner_kknn$train(task_scite, row_ids = train_set)
prediction_kknn = learner_kknn$predict(task_scite, row_ids = test_set)
prediction_kknn$confusion
autoplot(prediction_kknn)

# Training rpart learner ------

tune_tree = ParamSet$new(list(
  ParamDbl$new("cp", lower = 0.001, upper = 0.2),
  ParamInt$new("minsplit", lower = 10, upper = 50),
  ParamInt$new("minbucket", lower = 5, upper = 50)
))

learner = lrn("classif.rpart")
tuner = tnr("random_search")
measure = lapply(c("classif.ce", "classif.acc"), msr)
resample = rsmp("cv", folds = 6L)
resample$instantiate(task_scite)
evals40 = term("evals", n_evals = 40)


instance_rpart = TuningInstance$new(
  task = task_scite,
  learner = learner,
  resampling = resample,
  measures = measure,
  param_set = tune_tree,
  terminator = evals40
)

tuner$tune(instance_rpart)

learner$param_set$values = instance_rpart$result$params
learner$train(task_scite, row_ids = train_set)
prediction_rpart = learner$predict(task_scite, test_set)
prediction_rpart$confusion

autoplot(prediction_rpart)
prediction$confusion

sum((prediction$truth == prediction$response))

confusion_matrix <- read_xlsx("confusion_matrix.xlsx")
view(confusion_matrix)
stargazer(confusion_matrix, summary = FALSE)

subject_order <- c("math", "physics", "chemistry", "biology", 
                   "economics", "business", "medicine")

subject_order_reclassified <- c("math", "science",
                                "social_science", "medicine")

g <- data_reclassified %>%
  ggplot(aes(x = subject, y = sji_all_years)) +
  scale_x_discrete(limits = subject_order_reclassified) +
  geom_jitter(aes(col = subject), width = 0.30) +
  scale_color_economist() +
  theme(legend.position = "none")+
  labs(title = "SJI by Subject",
       x = "Subject", y = "SJI")
g

df1 <- data %>%
  mutate(percent_supporting = total_supporting_cites/total_cites*100) %>%
  mutate(percent_contradicting = total_contradicting_cites/total_cites*100) %>%
  select(sji_all_years, total_cites, percent_supporting, 
         percent_contradicting, subject)
view(df1)

# Show percent contradicting
df1 %>%
  filter(subject != "other") %>%
  ggplot(aes(x = subject, y = percent_contradicting)) +
  geom_jitter(aes(col = subject), width = 0.2) +
  scale_x_discrete(limits = subject_order) +
  scale_color_economist() +
  theme_economist()+
  theme(legend.position = "none")+
  labs(title = "Percent Contradicting Citations by Subject",
       x = "Subject", y = "Percent Contradicting Citations")

df1 %>%
  filter(subject != "other") %>%
  ggplot(aes(x = subject, y = percent_supporting)) +
  geom_jitter(aes(color = subject), width = 0.2) +
  scale_x_discrete(limits = subject_order) +
  scale_color_economist() +
  theme_economist() +
  theme(legend.position = "none")+
  labs(title = "Percent Supporting Citations by Subject",
       x = "Subject", y = "Percent Supporting Citations")

model1 <- lm(sji_all_years ~ total_cites + subject, data = data)
summary(model1)

stargazer(as.data.frame(data), 
          summary.stat = c("n", "mean", "sd", "min", "max"))
stargazer(model1)

# Model with reclassified data -----------
#

model_reclassified <- lm(sji_all_years ~ total_cites + subject, data = data_reclassified)
summary(model_reclassified)

confusion_matrix_reclassified <- read_xlsx("confusion_matrix.xlsx", sheet = 2)
view(confusion_matrix_reclassified)
# stargazer(confusion_matrix_reclassified, summary = FALSE)

stargazer(model_reclassified)
