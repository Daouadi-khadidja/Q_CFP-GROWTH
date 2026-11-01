# %% [markdown]
# # Cleaning the diabetes dataset for Apriori

# %% [markdown]
# ## 1. Load the data

# %%
import os
import numpy as np
import pandas as pd

# folder with diabetic_data.csv (works if the notebook is in the main folder or in a sub-folder)
DATA_DIR = "Data" if os.path.exists("Data") else "../Data"

# "?" = missing value. keep_default_na=False keeps the text "None" (= test not done)
df = pd.read_csv(DATA_DIR + "/diabetic_data.csv", keep_default_na=False, na_values=["?"])
print(df.shape)
df.head()

# %% [markdown]
# ## 2. Missing values

# %%
missing = df.isna().sum()
print(missing[missing > 0])

# %% [markdown]
# ## 3. Clean

# %% [markdown]
# **3.1 Drop columns with too many missing values** (`weight` is 97% missing, `payer_code` 40%)

# %%
df = df.drop(columns=["weight", "payer_code"])

# %% [markdown]
# **3.2 Remove invalid rows**
# - gender not valid
# - patients who died or went to hospice (they can not be readmitted, codes from `IDS_mapping.csv`)
# - no diagnosis at all

# %%
df = df[df["gender"] != "Unknown/Invalid"]

died_codes = [11, 13, 14, 19, 20, 21]
df = df[~df["discharge_disposition_id"].isin(died_codes)]

df = df.dropna(subset=["diag_1"])
print(df.shape)

# %% [markdown]
# **3.3 Keep only the first visit of each patient**
# The same patient appears many times, which would distort the patterns.
# We sort by `encounter_id` first, so "first" really means the earliest visit.

# %%
df = df.sort_values("encounter_id")
df = df.drop_duplicates(subset="patient_nbr", keep="first")
df = df.drop(columns=["encounter_id", "patient_nbr"])
print(df.shape)

# %% [markdown]
# **3.4 Drop medicine columns that are almost always `No`** (they give no useful pattern)

# %%
drug_cols = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
             "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
             "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide",
             "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
             "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]

kept_drugs = []
for drug in drug_cols:
    share_no = (df[drug] == "No").mean()     # share of rows with "No"
    if share_no >= 0.995:
        df = df.drop(columns=[drug])         # almost always "No" -> remove
    else:
        kept_drugs.append(drug)

print("Drugs kept:", kept_drugs)

# %% [markdown]
# ## 4. Group the ID columns (using `IDS_mapping.csv`)
# The groups get a clear name directly, so every item can be read on its own.

# %%
# Codes that mean NULL / Not Mapped / Unknown are not in the dictionaries -> they become NaN (= no item)
admission_type_dict = {1: "Emergency admission", 2: "Emergency admission", 7: "Emergency admission",
                       3: "Elective admission", 4: "Newborn admission"}

discharge_dict = {1: "Discharged home", 6: "Discharged home", 8: "Discharged home",
                  2: "Discharged to another facility", 3: "Discharged to another facility",
                  4: "Discharged to another facility", 5: "Discharged to another facility",
                  15: "Discharged to another facility", 16: "Discharged to another facility",
                  17: "Discharged to another facility", 22: "Discharged to another facility",
                  23: "Discharged to another facility", 24: "Discharged to another facility",
                  27: "Discharged to another facility", 28: "Discharged to another facility",
                  29: "Discharged to another facility", 30: "Discharged to another facility",
                  7: "Left against medical advice"}

admission_source_dict = {1: "Admitted by referral", 2: "Admitted by referral", 3: "Admitted by referral",
                         4: "Admitted by transfer", 5: "Admitted by transfer", 6: "Admitted by transfer",
                         10: "Admitted by transfer", 18: "Admitted by transfer", 22: "Admitted by transfer",
                         25: "Admitted by transfer", 26: "Admitted by transfer",
                         7: "Admitted from emergency room",
                         11: "Birth-related admission", 12: "Birth-related admission",
                         13: "Birth-related admission", 14: "Birth-related admission",
                         23: "Birth-related admission", 24: "Birth-related admission"}

df["admission_type"]   = df["admission_type_id"].map(admission_type_dict)
df["discharge"]        = df["discharge_disposition_id"].map(discharge_dict)
df["admission_source"] = df["admission_source_id"].map(admission_source_dict)

df = df.drop(columns=["admission_type_id", "discharge_disposition_id", "admission_source_id"])
df[["admission_type", "discharge", "admission_source"]].head()

# %% [markdown]
# **Medical specialty:** keep the 10 most frequent, the others become `Other`.

# %%
top10 = df["medical_specialty"].value_counts().head(10).index

# change only rows that are not missing and not in the top 10
mask = df["medical_specialty"].notna() & ~df["medical_specialty"].isin(top10)
df.loc[mask, "medical_specialty"] = "Other"

df["medical_specialty"].value_counts()

# %% [markdown]
# ## 5. Group the diagnosis codes (ICD-9)
# There are hundreds of codes, so we group them into 9 families.

# %%
def diagnosis_group(code):
    if pd.isna(code):
        return np.nan                      # keep missing as missing
    code = str(code)
    if code.startswith("V") or code.startswith("E"):
        return "Other"
    number = float(code)
    if 250 <= number < 251:
        return "Diabetes"
    elif 390 <= number <= 459 or number == 785:
        return "Circulatory"
    elif 460 <= number <= 519 or number == 786:
        return "Respiratory"
    elif 520 <= number <= 579 or number == 787:
        return "Digestive"
    elif 800 <= number <= 999:
        return "Injury"
    elif 710 <= number <= 739:
        return "Musculoskeletal"
    elif 580 <= number <= 629 or number == 788:
        return "Genitourinary"
    elif 140 <= number <= 239:
        return "Neoplasms"
    else:
        return "Other"

for col in ["diag_1", "diag_2", "diag_3"]:
    df[col] = df[col].apply(diagnosis_group)

df["diag_1"].value_counts()

# %% [markdown]
# ## 6. Turn numbers into categories
# Apriori can not use numbers, so we cut them into groups with `pd.cut`.
# Each group gets a name that explains itself (no `column=value` needed).
#
# We work on a **copy** called `items`: you can re-run this cell as many times as you want without errors.

# %%
items = df.copy()

items["time_in_hospital"] = pd.cut(items["time_in_hospital"], bins=[0, 3, 7, 100],
                                   labels=["Short stay (1-3 days)", "Medium stay (4-7 days)",
                                           "Long stay (8+ days)"]).astype(object)

items["num_lab_procedures"] = pd.cut(items["num_lab_procedures"], bins=[-1, 30, 50, 1000],
                                     labels=["Few lab tests (up to 30)", "Moderate lab tests (31-50)",
                                             "Many lab tests (51+)"]).astype(object)

items["num_procedures"] = pd.cut(items["num_procedures"], bins=[-1, 0, 2, 100],
                                 labels=["No procedures", "Few procedures (1-2)",
                                         "Many procedures (3+)"]).astype(object)

items["num_medications"] = pd.cut(items["num_medications"], bins=[-1, 10, 20, 1000],
                                  labels=["Few medications (up to 10)", "Moderate medications (11-20)",
                                          "Many medications (21+)"]).astype(object)

items["number_diagnoses"] = pd.cut(items["number_diagnoses"], bins=[-1, 5, 8, 100],
                                   labels=["Few diagnoses (up to 5)", "Moderate diagnoses (6-8)",
                                           "Many diagnoses (9+)"]).astype(object)

# previous visits: 0 / 1 / 2 or more
visit_names = {"number_outpatient": "outpatient", "number_emergency": "emergency",
               "number_inpatient": "inpatient"}
for col, kind in visit_names.items():
    items[col] = pd.cut(items[col], bins=[-1, 0, 1, 1000],
                        labels=[f"No previous {kind} visits", f"One previous {kind} visit",
                                f"Several previous {kind} visits"]).astype(object)

items[["time_in_hospital", "num_medications"]].head()

# %% [markdown]
# ## 7. Remove useless items
# - Drug value `No` = not prescribed → no item.
# - `None` = test not done → no item.

# %%
for drug in kept_drugs:
    items[drug] = items[drug].replace("No", np.nan)

items["A1Cresult"]     = items["A1Cresult"].replace("None", np.nan)
items["max_glu_serum"] = items["max_glu_serum"].replace("None", np.nan)

items.head()

# %% [markdown]
# ## 8. Give every item a clear, self-explanatory name
# No `column=value` any more: each item is a readable name such as
# `AfricanAmerican`, `Young adult (20-39)`, `Short stay (1-3 days)` or `Insulin dose increased`.

# %%
# --- race / gender ---
items["race"] = items["race"].replace({"Other": "OtherRace"})   # "Other" is used in other columns too

# --- age: 10 small groups -> 5 named life stages ---
age_groups = {
    "[0-10)": "Child/Teen (0-19)",      "[10-20)": "Child/Teen (0-19)",
    "[20-30)": "Young adult (20-39)",   "[30-40)": "Young adult (20-39)",
    "[40-50)": "Middle-aged (40-59)",   "[50-60)": "Middle-aged (40-59)",
    "[60-70)": "Senior (60-79)",        "[70-80)": "Senior (60-79)",
    "[80-90)": "Elderly (80+)",         "[90-100)": "Elderly (80+)",
}
items["age"] = items["age"].map(age_groups)

# --- lab tests ---
items["A1Cresult"] = items["A1Cresult"].replace(
    {"Norm": "HbA1c normal", ">7": "HbA1c above 7%", ">8": "HbA1c above 8%"})
items["max_glu_serum"] = items["max_glu_serum"].replace(
    {"Norm": "Glucose serum normal", ">200": "Glucose serum above 200", ">300": "Glucose serum above 300"})

# --- treatment ---
items["change"] = items["change"].replace(
    {"Ch": "Medication changed", "No": "Medication not changed"})
items["diabetesMed"] = items["diabetesMed"].replace(
    {"Yes": "Diabetes medication prescribed", "No": "No diabetes medication prescribed"})

# --- readmission (target) ---
items["readmitted"] = items["readmitted"].replace(
    {"<30": "Readmitted within 30 days", ">30": "Readmitted after 30 days", "NO": "Not readmitted"})

# --- diagnoses: say which diagnosis the family belongs to ---
for col, label in {"diag_1": "primary", "diag_2": "secondary", "diag_3": "additional"}.items():
    items[col] = items[col].map(lambda v, label=label: f"{v} ({label} diagnosis)" if pd.notna(v) else v)

# --- medical specialty: "InternalMedicine" -> "Internal Medicine (specialty)" ---
def specialty_name(value):
    if pd.isna(value):
        return value
    value = pd.Series([value]).str.replace(r"(?<=[a-z])(?=[A-Z])", " ", regex=True)[0]
    return value.replace("/", " / ") + " (specialty)"

items["medical_specialty"] = items["medical_specialty"].map(specialty_name)

# --- drugs: "Insulin" + "Up" -> "Insulin dose increased" ---
drug_names = {
    "metformin": "Metformin", "repaglinide": "Repaglinide", "nateglinide": "Nateglinide",
    "chlorpropamide": "Chlorpropamide", "glimepiride": "Glimepiride", "acetohexamide": "Acetohexamide",
    "glipizide": "Glipizide", "glyburide": "Glyburide", "tolbutamide": "Tolbutamide",
    "pioglitazone": "Pioglitazone", "rosiglitazone": "Rosiglitazone", "acarbose": "Acarbose",
    "miglitol": "Miglitol", "troglitazone": "Troglitazone", "tolazamide": "Tolazamide",
    "examide": "Examide", "citoglipton": "Citoglipton", "insulin": "Insulin",
    "glyburide-metformin": "Glyburide + Metformin", "glipizide-metformin": "Glipizide + Metformin",
    "glimepiride-pioglitazone": "Glimepiride + Pioglitazone",
    "metformin-rosiglitazone": "Metformin + Rosiglitazone",
    "metformin-pioglitazone": "Metformin + Pioglitazone",
}
dose_text = {"Steady": "dose unchanged", "Up": "dose increased", "Down": "dose decreased"}
for drug in kept_drugs:
    name = drug_names[drug]
    items[drug] = items[drug].map(lambda v, name=name: f"{name} {dose_text[v]}" if pd.notna(v) else v)

items.head()

# %% [markdown]
# ## 9. One-hot encoding (True/False table for Apriori)

# %%
# every value becomes one column of 1 / 0 named by the value itself ; missing values give no item
parts = [pd.get_dummies(items[col], prefix="", prefix_sep="", dtype=int) for col in items.columns]
onehot = pd.concat(parts, axis=1)

# every item name must be unique (same name in two columns would mix two different things)
assert not onehot.columns.duplicated().any(), onehot.columns[onehot.columns.duplicated()].tolist()

# remove very rare items (less than 0.5% of patients): they can never be frequent
support = onehot.mean()
rare_items = support[support < 0.005].index
onehot = onehot.drop(columns=rare_items)

print("Removed", len(rare_items), "rare items")
print("Final table:", onehot.shape)
onehot.head()

# %% [markdown]
# ## 10. Quick check

# %%
print("Patients  :", onehot.shape[0])
print("Items     :", onehot.shape[1])
print("Items per patient (average):", round(onehot.sum(axis=1).mean(), 1))
print("Empty patients:", (onehot.sum(axis=1) == 0).sum())

# most frequent items
onehot.mean().sort_values(ascending=False).head(10).round(3)

# %% [markdown]
# ## 11. Save the Apriori file

# %%
onehot.to_csv(DATA_DIR + "/diabetic_data_apriori.csv", index=False)
print("Saved:", DATA_DIR + "/diabetic_data_apriori.csv", onehot.shape)