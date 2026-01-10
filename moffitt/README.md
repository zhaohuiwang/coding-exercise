
## Data:

### CDC Wonder API
  - Total counts of cancer occurrences for 2014-2018 for three major cancer types: Breast, Lung & Bronchus, Melanoma of the Skin
  - Data provides counts of each cancer type within each state over a 5 year period, with separate counts for males and females as well as for three distinct age groups. 
  - Three files, split by age groups:
    - `data/cancer_count_by_state_year_sex_agegte25lt44.txt` (ages of 25 - 44 years)
    - `data/cancer_count_by_state_year_sex_agegte45lt65.txt` (ages of 45 - 64 years)
    - `data/cancer_count_by_state_year_sex_agegte65.txt` (ages of 65+ years)
  - Source [documentation](https://wonder.cdc.gov/wonder/help/cancer-v2018.html)

### Population/Economic Indicators
  - In a SQLite database `data/population_data.db`
  - This database has two tables:
    - `population`: contains the annual population of each state from 2010 to 2020, by sex and age group (sourced from census.gov, [link](https://www2.census.gov/programs-surveys/popest/datasets/2010-2020/state/asrh/SC-EST2020-AGESEX-CIV.csv))
    - `economic_indicators`: contains several economic indicators compiled by the Bureau of Economic Analysis, by state and year (sourced from bea.gov, [link](https://www.bea.gov/data/income-saving/personal-income-county-metro-and-other-areas))

### Smoking Rate Data:
  - Yearly survey data on cigarette smoking rates by state.
  - One file (json):
    - `data/cdc_smoking_data.json`
  - [Source](https://chronicdata.cdc.gov/Survey-Data/Behavioral-Risk-Factor-Data-Tobacco-Use-2011-to-pr/wsas-xwh5)

