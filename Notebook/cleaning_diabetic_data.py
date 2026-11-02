# %% [markdown]
# # Cleaning the diabetes dataset for Apriori

# %% [markdown]
# ## 1. Load the data

# %%
import os
import re
import numpy as np
import pandas as pd

# Find the folder that contains the data
if os.path.exists("Data"):
    DATA_DIR = "Data"
else:
    DATA_DIR = "../Data"

# "?" means missing value, and the text "None" is kept (it means test not done)
df = pd.read_csv(DATA_DIR + "/diabetic_data.csv", keep_default_na=False, na_values=["?"])
print(df.shape)
df.head()

# %% [markdown]
# ## 2. Missing values

# %%
# Count the missing values in each column and show only the columns that have some
missing = df.isna().sum()
print(missing[missing > 0])

# %% [markdown]
# ## 3. Clean the data

# %% [markdown]
# ### 3.1 Drop the columns with too many missing values

# %%
# weight is 97% missing and payer_code is 40% missing
df = df.drop(columns=["weight", "payer_code"])

# %% [markdown]
# ### 3.2 Remove the invalid rows

# %%
# Remove rows with an invalid gender
df = df[df["gender"] != "Unknown/Invalid"]

# Remove patients who died or went to hospice (they can not be readmitted)
died_codes = [11, 13, 14, 19, 20, 21]
df = df[~df["discharge_disposition_id"].isin(died_codes)]

# Remove rows with no diagnosis at all
df = df.dropna(subset=["diag_1"])
print(df.shape)

# %% [markdown]
# ### 3.3 Keep only the first visit of each patient

# %%
# Sort by encounter_id so the first row of a patient is his earliest visit
df = df.sort_values("encounter_id")

# Keep only one row per patient (the first one)
df = df.drop_duplicates(subset="patient_nbr", keep="first")

# We do not need the two id columns anymore
df = df.drop(columns=["encounter_id", "patient_nbr"])
print(df.shape)

# %% [markdown]
# ### 3.4 Drop the medicine columns that are almost always "No"

# %%
drug_cols = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
             "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
             "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide",
             "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
             "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]

kept_drugs = []
for drug in drug_cols:
    # Share of patients who have "No" for this drug
    share_no = (df[drug] == "No").mean()
    if share_no >= 0.995:
        df = df.drop(columns=[drug])
    else:
        kept_drugs.append(drug)

print("Drugs kept:", kept_drugs)

# %% [markdown]
# ## 4. Group the ID columns

# %%
# Each group name is followed by the list of codes that belong to it
# (codes that are not in the lists, like NULL or Unknown, become NaN = no item)
admission_type_groups = {
    "Emergency admission": [1, 2, 7],
    "Elective admission": [3],
    "Newborn admission": [4],
}

discharge_groups = {
    "Discharged home": [1, 6, 8],
    "Discharged to another facility": [2, 3, 4, 5, 15, 16, 17, 22, 23, 24, 27, 28, 29, 30],
    "Left against medical advice": [7],
}

admission_source_groups = {
    "Admitted by referral": [1, 2, 3],
    "Admitted by transfer": [4, 5, 6, 10, 18, 22, 25, 26],
    "Admitted from emergency room": [7],
    "Birth-related admission": [11, 12, 13, 14, 23, 24],
}


def find_group(code, groups):
    """Give the name of the group that contains this code (NaN if none)."""
    for group_name, codes in groups.items():
        if code in codes:
            return group_name
    return np.nan


# Create the 3 new columns with the group names
df["admission_type"] = df["admission_type_id"].apply(find_group, groups=admission_type_groups)
df["discharge"] = df["discharge_disposition_id"].apply(find_group, groups=discharge_groups)
df["admission_source"] = df["admission_source_id"].apply(find_group, groups=admission_source_groups)

# Remove the old id columns
df = df.drop(columns=["admission_type_id", "discharge_disposition_id", "admission_source_id"])
df[["admission_type", "discharge", "admission_source"]].head()

# %% [markdown]
# ### Medical specialty: keep the 10 most frequent

# %%
top10 = df["medical_specialty"].value_counts().head(10).index

# Rows that have a specialty which is not in the top 10 become "Other"
not_missing = df["medical_specialty"].notna()
not_in_top10 = ~df["medical_specialty"].isin(top10)
df.loc[not_missing & not_in_top10, "medical_specialty"] = "Other"

df["medical_specialty"].value_counts()

# %% [markdown]
# ## 5. Group the diagnosis codes (ICD-9)

# %%
def diagnosis_group(code):
    """Give the family of a diagnosis code (9 families in total)."""
    if pd.isna(code):
        return np.nan

    code = str(code)

    # Codes starting with V or E are not numbers
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


# Apply the function on the 3 diagnosis columns
for col in ["diag_1", "diag_2", "diag_3"]:
    df[col] = df[col].apply(diagnosis_group)

df["diag_1"].value_counts()

# %% [markdown]
# ## 6. Turn numbers into categories

# %%
# Work on a copy, so this cell can be run again without errors
items = df.copy()

# Days in hospital
items["time_in_hospital"] = pd.cut(
    items["time_in_hospital"], bins=[0, 3, 7, 100],
    labels=["Short stay (1-3 days)", "Medium stay (4-7 days)", "Long stay (8+ days)"]).astype(object)

# Number of lab tests
items["num_lab_procedures"] = pd.cut(
    items["num_lab_procedures"], bins=[-1, 30, 50, 1000],
    labels=["Few lab tests (up to 30)", "Moderate lab tests (31-50)", "Many lab tests (51+)"]).astype(object)

# Number of procedures
items["num_procedures"] = pd.cut(
    items["num_procedures"], bins=[-1, 0, 2, 100],
    labels=["No procedures", "Few procedures (1-2)", "Many procedures (3+)"]).astype(object)

# Number of medications
items["num_medications"] = pd.cut(
    items["num_medications"], bins=[-1, 10, 20, 1000],
    labels=["Few medications (up to 10)", "Moderate medications (11-20)", "Many medications (21+)"]).astype(object)

# Number of diagnoses
items["number_diagnoses"] = pd.cut(
    items["number_diagnoses"], bins=[-1, 5, 8, 100],
    labels=["Few diagnoses (up to 5)", "Moderate diagnoses (6-8)", "Many diagnoses (9+)"]).astype(object)

# Previous visits: 0, 1, or 2 and more
visit_names = {"number_outpatient": "outpatient",
               "number_emergency": "emergency",
               "number_inpatient": "inpatient"}

for col, kind in visit_names.items():
    items[col] = pd.cut(
        items[col], bins=[-1, 0, 1, 1000],
        labels=["No previous " + kind + " visits",
                "One previous " + kind + " visit",
                "Several previous " + kind + " visits"]).astype(object)

items[["time_in_hospital", "num_medications"]].head()

# %% [markdown]
# ## 7. Remove the useless items

# %%
# Drug "No" means not prescribed, so it is not an item
for drug in kept_drugs:
    items[drug] = items[drug].replace("No", np.nan)

# "None" means the test was not done, so it is not an item
items["A1Cresult"] = items["A1Cresult"].replace("None", np.nan)
items["max_glu_serum"] = items["max_glu_serum"].replace("None", np.nan)

items.head()

# %% [markdown]
# ## 8. Give every item a clear name

# %% [markdown]
# ### Race

# %%
# "Other" is already used in other columns, so we rename this one
items["race"] = items["race"].replace({"Other": "OtherRace"})

# %% [markdown]
# ### Age (10 small groups become 5 life stages)

# %%
age_groups = {
    "[0-10)": "Child/Teen (0-19)",    "[10-20)": "Child/Teen (0-19)",
    "[20-30)": "Young adult (20-39)", "[30-40)": "Young adult (20-39)",
    "[40-50)": "Middle-aged (40-59)", "[50-60)": "Middle-aged (40-59)",
    "[60-70)": "Senior (60-79)",      "[70-80)": "Senior (60-79)",
    "[80-90)": "Elderly (80+)",       "[90-100)": "Elderly (80+)",
}
items["age"] = items["age"].map(age_groups)

# %% [markdown]
# ### Lab tests

# %%
items["A1Cresult"] = items["A1Cresult"].replace(
    {"Norm": "HbA1c normal", ">7": "HbA1c above 7%", ">8": "HbA1c above 8%"})

items["max_glu_serum"] = items["max_glu_serum"].replace(
    {"Norm": "Glucose serum normal", ">200": "Glucose serum above 200", ">300": "Glucose serum above 300"})

# %% [markdown]
# ### Treatment

# %%
items["change"] = items["change"].replace(
    {"Ch": "Medication changed", "No": "Medication not changed"})

items["diabetesMed"] = items["diabetesMed"].replace(
    {"Yes": "Diabetes medication prescribed", "No": "No diabetes medication prescribed"})

# %% [markdown]
# ### Readmission (the target)

# %%
items["readmitted"] = items["readmitted"].replace(
    {"<30": "Readmitted within 30 days", ">30": "Readmitted after 30 days", "NO": "Not readmitted"})

# %% [markdown]
# ### Diagnoses (add which diagnosis it is)

# %%
diagnosis_labels = {"diag_1": "primary", "diag_2": "secondary", "diag_3": "additional"}

for col, label in diagnosis_labels.items():
    new_values = []
    for value in items[col]:
        if pd.notna(value):
            new_values.append(value + " (" + label + " diagnosis)")
        else:
            new_values.append(value)
    items[col] = new_values

# %% [markdown]
# ### Medical specialty ("InternalMedicine" becomes "Internal Medicine (specialty)")

# %%
def specialty_name(value):
    """Write the specialty in a readable way."""
    if pd.isna(value):
        return value

    # Add a space between a small letter and a capital letter
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    value = value.replace("/", " / ")
    return value + " (specialty)"


items["medical_specialty"] = items["medical_specialty"].map(specialty_name)

# %% [markdown]
# ### Drugs ("Insulin" + "Up" becomes "Insulin dose increased")

# %%
dose_text = {"Steady": "dose unchanged", "Up": "dose increased", "Down": "dose decreased"}

for drug in kept_drugs:
    # "glyburide-metformin" becomes "Glyburide + Metformin"
    name = drug.replace("-", " + ").title()

    new_values = []
    for value in items[drug]:
        if pd.notna(value):
            new_values.append(name + " " + dose_text[value])
        else:
            new_values.append(value)
    items[drug] = new_values

items.head()

# %% [markdown]
# ## 9. One-hot encoding (1/0 table for Apriori)

# %%
# Each value becomes one column of 1 and 0 (missing values give no item)
parts = []
for col in items.columns:
    one_column = pd.get_dummies(items[col], prefix="", prefix_sep="", dtype=int)
    parts.append(one_column)

onehot = pd.concat(parts, axis=1)

# Check that every item name is unique
assert not onehot.columns.duplicated().any(), onehot.columns[onehot.columns.duplicated()].tolist()

# Remove the very rare items (less than 0.5% of patients), they can never be frequent
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

# The 10 most frequent items
onehot.mean().sort_values(ascending=False).head(10).round(3)

# %% [markdown]
# ## 11. Save the Apriori file

# %%
onehot.to_csv(DATA_DIR + "/diabetic_data_apriori.csv", index=False)
print("Saved:", DATA_DIR + "/diabetic_data_apriori.csv", onehot.shape)