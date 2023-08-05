# Preprocessing -----------------------------------------------------------

library(tidyverse)
library(lme4)
library(report)
library(xtable)

results_blanco <- read_csv2("G:/Meine Ablage/Studium/Masterarbeit/auswertung_studie/results/results_blanco.csv") %>% 
  mutate(condition = "no_exp")

results_explanation <- read_csv2("G:/Meine Ablage/Studium/Masterarbeit/auswertung_studie/results/results_expl.csv") %>% 
  mutate(condition = "exp")

results <- rbind(results_blanco, results_explanation)

results_cleaned <- results %>% 
  filter(`# ResponseStatus` == "Komplette Antwort") %>% 
  select(-1, -c(39:45)) 

colnames(results_cleaned) <-  c(
  "q01_correct",
  "q01_estimated_accuracy",
  "q02_correct",
  "q02_estimated_accuracy",
  "q03_correct",
  "q03_estimated_accuracy",
  "q04_correct",
  "q04_estimated_accuracy",
  "q05_correct",
  "q05_estimated_accuracy",
  "q06_correct",
  "q06_estimated_accuracy",
  "q07_correct",
  "q07_estimated_accuracy",
  "q08_correct",
  "q08_estimated_accuracy",
  "q09_correct",
  "q09_estimated_accuracy",
  "q10_correct",
  "q10_estimated_accuracy",
  # Items 4 and 8 where removed
  "quest_01",
  "quest_02",
  "quest_03",
  "quest_05",
  "quest_06",
  "quest_07",
  "quest_09",
  "quest_10",
  "quest_11",
  "quest_12",
  "quest_13",
  "quest_14",
  "quest_15",
  "quest_16",
  "quest_17",
  "quest_18",
  "quest_19",
  "condition")

# Test Correctness --------------------------------------------------------

# https://stats.oarc.ucla.edu/other/mult-pkg/introduction-to-generalized-linear-mixed-models/
results_cleaned_correct <- results_cleaned %>% 
  select(contains("correct"), "condition") %>% 
  mutate(across(q01_correct:q10_correct, ~if_else(.=="Ja", 1, 0))) %>% 
  rownames_to_column("id") %>% 
  pivot_longer(!c("id", "condition"), names_to = "q_id", values_to = "q_correct") %>% 
  mutate(q_id = substr(q_id, start = 1, stop = 3))
  
results_cleaned_correct_vis <- results_cleaned_correct %>% 
  group_by(condition, q_id) %>% 
  summarise(q_correct = mean(q_correct)) %>% 
  ungroup() %>% 
  mutate(q_correct = paste(as.character(round(q_correct*100,1)),"%")) %>% 
  pivot_wider(names_from = "condition", values_from = q_correct)
  
print(xtable(results_cleaned_correct_vis), type = "latex", include.rownames = FALSE)

m1 <- glmer(q_correct ~ q_id + condition + (1|id), data = results_cleaned_correct, family = binomial, 
             control = glmerControl(optimizer = "bobyqa"), nAGQ = 10)
summary(m1)
report(m1)

# Test Accuracy -----------------------------------------------------------

perc_to_num <- function(inp){
  temp <- gsub("%","", inp) %>% 
    as.numeric()
  return(temp)
}

results_cleaned_accuracy <- results_cleaned %>% 
  select(contains("estimated_accuracy"), "condition") %>% 
  mutate(across(q01_estimated_accuracy:q10_estimated_accuracy, ~perc_to_num(.))) %>% 
  rownames_to_column("id") %>% 
  pivot_longer(!c("id", "condition"), names_to = "q_id", values_to = "q_estimated_accuracy") %>% 
  mutate(q_id = substr(q_id, start = 1, stop = 3))

results_cleaned_accuracy_mean <- results_cleaned_accuracy %>% 
  group_by(condition, q_id) %>% 
  summarise(q_estimated_accuracy = mean(q_estimated_accuracy))

acc_plot <- ggplot(results_cleaned_accuracy, aes(x = condition, y = q_estimated_accuracy, fill = condition)) + 
  geom_boxplot() +
  facet_grid(~q_id) +
  coord_cartesian(ylim = c(0,100)) +
  scale_fill_discrete(name = "", labels = c("Explantions", "No Explantions")) +
  labs(y = "Estimated Accuracy") +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.title.x = element_blank())

ggsave(filename = "acc_plot.png", acc_plot, width = 7, height = 3, dpi = 300)

m2 <- lmer(q_estimated_accuracy ~ q_id + condition + (1|id), data=results_cleaned_accuracy, REML = FALSE)
summary(m2)
report(m2)

# Test Accuracy + Correct -------------------------------------------------

results_cleaned_accuracycorrect <- left_join(
  results_cleaned_accuracy,
  results_cleaned_correct,
  by = c("id"="id", "condition"="condition", "q_id" = "q_id")) %>% 
  mutate(q_correct = as.character(q_correct))

results_cleaned_accuracycorrect_mean <- results_cleaned_accuracycorrect %>% 
  group_by(q_id, condition, q_correct) %>% 
  summarise(q_estimated_accuracy = mean(q_estimated_accuracy, na.rm = T))

acc_correct_plot <- ggplot(
  results_cleaned_accuracycorrect %>% 
    mutate(q_correct = if_else(q_correct == 1, "model decision correct", "model decision incorrect")),
           aes(x = 1, y = q_estimated_accuracy, fill = condition)) + 
  geom_boxplot(width = .5, position = position_dodge()) +
  geom_vline(xintercept = 1.5, size = 3, color = "white") +
  facet_grid(vars(q_correct), vars(q_id)) +
  coord_cartesian(ylim = c(0,100)) +
  scale_fill_discrete(name = "", labels = c("Explantions", "No Explantions")) +
  labs(y = "Estimated Accuracy") +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.title.x = element_blank())

ggsave(filename = "acc_correct_plot.png", acc_correct_plot, width = 7, height = 6, dpi = 300)

m3 <- lmer(q_estimated_accuracy ~ q_id + condition*q_correct + (1|id), data=results_cleaned_accuracycorrect, REML = FALSE)

summary(m3)
report(m3)

prediction_visualized_m3 <- results_cleaned_accuracycorrect %>% 
  mutate(prediction = predict(m3, newdata = results_cleaned_accuracycorrect, allow.new.levels = TRUE))

acc_correct_pred_plot <- ggplot(prediction_visualized_m3, aes(x = q_correct, y = prediction, fill = condition)) + 
  #geom_point() +
  geom_boxplot(width = .5, position = position_dodge()) +
  geom_vline(xintercept = 1.5, size = 3, color = "white") +
  facet_grid(~q_id) +
  coord_cartesian(ylim = c(0,100)) +
  scale_color_discrete(name = "", labels = c("Explantions", "No Explantions")) +
  labs(x = "Model decision correct?", y = "Predicted Estimated Accuracy")

ggsave(filename = "acc_correct_pred_plot.png", acc_correct_plot, width = 9, height = 3, dpi = 300)

fixed_effects <- coef(summary(m3))

main_effects_m3 <-  results_cleaned_accuracycorrect %>% 
  group_by(q_id, condition, q_correct) %>% 
  summarise() %>% 
  ungroup() %>% 
  mutate(prediction = 
           case_when(
             q_id == "q01" ~ fixed_effects[1,1],
             q_id == "q02" ~ fixed_effects[1,1] + fixed_effects[2,1],
             q_id == "q03" ~ fixed_effects[1,1] + fixed_effects[3,1],
             q_id == "q04" ~ fixed_effects[1,1] + fixed_effects[4,1],
             q_id == "q05" ~ fixed_effects[1,1] + fixed_effects[5,1],
             q_id == "q06" ~ fixed_effects[1,1] + fixed_effects[6,1],
             q_id == "q07" ~ fixed_effects[1,1] + fixed_effects[7,1],
             q_id == "q08" ~ fixed_effects[1,1] + fixed_effects[8,1],
             q_id == "q09" ~ fixed_effects[1,1] + fixed_effects[9,1],
             q_id == "q10" ~ fixed_effects[1,1] + fixed_effects[10,1],
             TRUE ~ -Inf),
         prediction = if_else(condition == "no_exp", prediction + fixed_effects[11,1], prediction),
         prediction = if_else(q_correct == 1, prediction + fixed_effects[12,1], prediction),
         prediction = if_else(condition == "no_exp" & q_correct == 1, prediction + fixed_effects[13,1], prediction))

ggplot(main_effects_m3, aes(x = q_correct, y = prediction , color = condition)) +
  geom_point() +
  labs(x = "Model decision correct?", y = "Estimated Accuracy") +
  scale_color_discrete(name = "", labels = c("Explantions", "No Explantions")) +
  facet_grid(~q_id)

# Are their systematic difference between correctness and accuracy? -------
 
results_cleaned_difference_observed_estimated_accuracy <- results_cleaned %>% 
  select(contains("correct"), contains("estimated_accuracy"), "condition") %>%
  mutate(across(q01_correct:q10_correct, ~if_else(.=="Ja", 1, 0))) %>% 
  mutate(across(q01_estimated_accuracy:q10_estimated_accuracy, ~perc_to_num(.))) %>% 
  mutate(
    q01_observedacc_estacc_dif = q02_estimated_accuracy - q01_correct * 100,
    q02_observedacc_estacc_dif = q02_estimated_accuracy - (rowSums(across(q01_correct:q02_correct))/2 * 100),
    q03_observedacc_estacc_dif = q03_estimated_accuracy - (rowSums(across(q01_correct:q03_correct))/3 * 100),
    q04_observedacc_estacc_dif = q04_estimated_accuracy - (rowSums(across(q01_correct:q04_correct))/4 * 100),
    q05_observedacc_estacc_dif = q05_estimated_accuracy - (rowSums(across(q01_correct:q05_correct))/5 * 100),
    q06_observedacc_estacc_dif = q06_estimated_accuracy - (rowSums(across(q01_correct:q06_correct))/6 * 100),
    q07_observedacc_estacc_dif = q07_estimated_accuracy - (rowSums(across(q01_correct:q07_correct))/7 * 100),
    q08_observedacc_estacc_dif = q08_estimated_accuracy - (rowSums(across(q01_correct:q08_correct))/8 * 100),
    q09_observedacc_estacc_dif = q09_estimated_accuracy - (rowSums(across(q01_correct:q09_correct))/9 * 100),
    q10_observedacc_estacc_dif = q10_estimated_accuracy - (rowSums(across(q01_correct:q10_correct))/10 * 100)) %>% 
  rownames_to_column("id") %>% 
  select(id, condition, q01_observedacc_estacc_dif:q10_observedacc_estacc_dif) %>% 
  pivot_longer(!c("id", "condition"), names_to = "q_id", values_to = "dif_obs_est_acc")

results_cleaned_difference_observed_estimated_accuracy_mean <- results_cleaned_difference_observed_estimated_accuracy %>% 
  group_by(q_id, condition) %>% 
  summarise(dif_obs_est_acc = mean(dif_obs_est_acc))

ggplot(
  results_cleaned_difference_observed_estimated_accuracy,
  aes(x = q_id, y=dif_obs_est_acc, fill = condition)) +
  geom_boxplot(position = position_dodge()) +
  labs(
    y = "Difference Estimated Accuracy - Oberserved Accuracy
    (+ Overestimated, - Underestimate)") +
  geom_hline(yintercept = 0, color = "red")

m3.1 <- lmer(dif_obs_est_acc ~ q_id + condition + (1|id), data=results_cleaned_difference_observed_estimated_accuracy, REML = FALSE)
summary(m3.1)
report(m3.1)
confint(m3.1)
# https://www.ssc.wisc.edu/sscc/pubs/MM/MM_DiagInfer.html

# Test Accuracy as an Effect of Error -------------------------------------

results_cleaned_change <- results_cleaned %>% 
  select(contains("correct"), contains("estimated_accuracy"), "condition") %>%
  mutate(across(q01_correct:q10_correct, ~if_else(.=="Ja", 1, 0))) %>% 
  mutate(across(q01_estimated_accuracy:q10_estimated_accuracy, ~perc_to_num(.))) %>% 
  mutate(q02_dif_to_prev_acc = q02_estimated_accuracy - q01_estimated_accuracy,
         q03_dif_to_prev_acc = q03_estimated_accuracy - q02_estimated_accuracy,
         q04_dif_to_prev_acc = q04_estimated_accuracy - q03_estimated_accuracy,
         q05_dif_to_prev_acc = q05_estimated_accuracy - q04_estimated_accuracy,
         q06_dif_to_prev_acc = q06_estimated_accuracy - q05_estimated_accuracy,
         q07_dif_to_prev_acc = q07_estimated_accuracy - q06_estimated_accuracy,
         q08_dif_to_prev_acc = q08_estimated_accuracy - q07_estimated_accuracy,
         q09_dif_to_prev_acc = q09_estimated_accuracy - q08_estimated_accuracy,
         q10_dif_to_prev_acc = q10_estimated_accuracy - q09_estimated_accuracy) %>% 
  select(contains("correct"), contains("dif_to_prev_acc"), "condition") %>% 
  rownames_to_column("id") %>% 
  pivot_longer(!c("id", "condition"), names_to = "var", values_to = "val") %>% 
  mutate(
    tp = substr(var, start = 1, stop = 3),
    measure = substr(var, start = 5, stop=nchar(var))) %>% 
  filter(tp != "q01") %>% 
  pivot_wider(c("id", "tp", "condition"), names_from = measure, values_from = val) %>% 
  mutate(
    correct_condition = case_when(
      condition == "exp" & correct == 1 ~ "exp_correct_pred",
      condition == "exp" & correct == 0 ~ "exp_wrong_pred",
      condition == "no_exp" & correct == 1 ~ "noexp_correct_pred",
      condition == "no_exp" & correct == 0 ~ "noexp_wrong_pred",
      TRUE ~ ""))

results_cleaned_change_mean <- results_cleaned_change %>% 
  group_by(condition, tp, correct) %>% 
  summarise(dif_to_prev_acc = mean(dif_to_prev_acc))

ggplot(results_cleaned_change_mean, aes(x = condition, y = dif_to_prev_acc, color = correct)) +
  geom_jitter(height = 0, width = .2, alpha = .5) +
  facet_grid(~ tp)

m4.0 <- lmer(dif_to_prev_acc ~ correct + (1|tp), data=results_cleaned_change, REML = FALSE)
m4.1 <- lmer(dif_to_prev_acc ~ correct_condition + (1|tp), data=results_cleaned_change, REML = FALSE)

anova(m4.0, m4.1)
summary(m4.1)

# Effects on Trust Subscales ----------------------------------------------

answer_to_number <- function(inp){
  temp <- if_else(
    inp == "Stimme gar nicht zu", 1, if_else(
      inp == "Stimme eher nicht zu", 2, if_else(
        inp == "Stimme weder zu noch nicht zu", 3, if_else(
          inp == "Stimme eher zu", 4, if_else(
            inp == "Stimme voll zu", 5, NaN)))))
  return(temp) 
}

results_questionaire <- results_cleaned %>% 
  select(contains("quest"), "condition") %>% 
  mutate(across(quest_01:quest_19, ~ answer_to_number(.))) %>% 
  rownames_to_column("id") %>% 
  pivot_longer(!c("id", "condition"), names_to = "q_id", values_to = "answer") %>% 
  mutate(
    # Adjust inverted items
    answer = if_else(q_id %in% c("quest_05", "quest_07", "quest_10", "quest_15", "quest_16"), - answer + 6, answer),
    subscale = case_when(
      str_detect(q_id, "quest_01|quest_06|quest_10|quest_13|quest_15|quest_19") ~ "R/K",
      str_detect(q_id, "quest_02|quest_07|quest_11|quest_16") ~ "V/V",
      str_detect(q_id, "quest_03|quest_17") ~ "Ve",
      str_detect(q_id, "quest_05|quest_12|quest_18") ~ "N",
      str_detect(q_id, "quest_09|quest_14") ~ "ViA",
      TRUE ~ "NaN")) %>% 
  group_by(id, condition, subscale) %>% 
  summarise(subscale_score = mean(answer, na.rm = TRUE))

results_questionaire_mean <- results_questionaire %>% 
  group_by(condition, subscale) %>% 
  summarise(subscale_score = mean(subscale_score, na.rm = TRUE))

results_questionaire_correct_acc <- left_join(
    results_questionaire,
    results_cleaned_correct %>%
      group_by(id, condition) %>%
      summarise(perc_correct = sum(q_correct)/10),
    by = c("id"="id", "condition"="condition")) %>% 
  left_join(
    results_cleaned_accuracy %>% 
      filter(q_id == "q10") %>% 
      select(!q_id),
    by = c("id"="id", "condition"="condition"))

tia_plot <- ggplot(
  results_questionaire %>% 
    filter(subscale %in% c("R/K", "V/V", "ViA", "N")) %>% 
    mutate(subscale = case_when(
      subscale == "R/K" ~ "Reliability/Competence",
      subscale == "V/V" ~ "Understanding/Predictability",
      subscale == "ViA" ~ "Trust in Automation",
      subscale == "N" ~ "Propensity to Trust",
      TRUE ~ "unknown name")),
  aes(x = 1, y = subscale_score, fill = condition)) +
  geom_boxplot() +
  labs(y = "mean subscale score") + 
  facet_grid(~subscale) +
  scale_fill_discrete(name = "", labels = c("Explantions", "No Explantions")) +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.title.x = element_blank(),
        strip.text.x = element_text(size = 6))

ggsave(filename = "tia_plot.png", tia_plot, width = 7, height = 3, dpi = 300)

results_questionaire_correct_acc_wide <- pivot_wider(results_questionaire_correct_acc, names_from = subscale, values_from = subscale_score)

m4.1 <- lm(data = results_questionaire_correct_acc_wide, `R/K` ~ condition*q_estimated_accuracy)
summary(m4.1)
report(m4.1)

m4.2 <- lm(data = results_questionaire_correct_acc_wide, `V/V` ~ condition + q_estimated_accuracy)
summary(m4.2)
report(m4.2)

m4.3 <- lm(data = results_questionaire_correct_acc_wide, `N` ~ condition + q_estimated_accuracy)
summary(m4.3)
report(m4.3)

m4.4 <- lm(data = results_questionaire_correct_acc_wide, ViA ~ N + `R/K` + `V/V` + condition)
summary(m4.4)
report(m4.4)

test <- ""
for (x in 1:1000) {
  test <- paste(test, "a ")
}




