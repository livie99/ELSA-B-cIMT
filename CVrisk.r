library(readr)
library(dplyr)
library(CVrisk)

BASE_ROOT <- file.path(getwd(), "data", "tab_202604")

train_df <- read_csv(file.path(BASE_ROOT, "train.csv"), show_col_types = FALSE)
val_df <- read_csv(file.path(BASE_ROOT, "valid.csv"), show_col_types = FALSE)
test_df <- read_csv(file.path(BASE_ROOT, "test.csv"), show_col_types = FALSE)

preprocess <- function(df) {
  
  df <- df %>%
    mutate(
      Sex = ifelse(.data$Sex == 1, "male", "female"),
      Smoking = ifelse(.data$Smoking == 2, TRUE, FALSE)

    )
  return(df)
}

## PREVENT
## age 30-79
## sbp 90-180, some
## totchol 130-320, some
## hdl 20-100, some
## egfr 15-140
## bmi 19-39, some
## uacr no limit
## hba1c no limit

train_df <- preprocess(train_df)
val_df <- preprocess(val_df)
test_df <- preprocess(test_df)


train_df <- train_df %>%
  rowwise() %>%
  mutate(
    frs = ascvd_10y_frs(
      gender = Sex,
      age = Age,
      hdl = HDL,
      totchol = Total_cholesterol,
      sbp = SBP,
      bp_med = Antihypertensive_med,
      smoker = Smoking,
      diabetes = Diabetes,
    )
  ) %>%
  mutate(
    prevent = ascvd_10y_prevent(
      gender = Sex,
      age = Age,
      sbp = SBP,
      bp_med = Antihypertensive_med,
      totchol = Total_cholesterol,
      hdl = HDL,
      statin = statin_use,
      diabetes = Diabetes,
      smoker = Smoking,
      egfr = eGFR,
      bmi = BMI,
      uacr = UACR,
      hba1c = HbA1c
    )
  ) %>%
  ungroup()

print(train_df$frs)
print(train_df$prevent)
