# %%
import json
from numpy.core.numeric import NaN
import pandas as pd
import numpy as np

# %%
animals = [15, 16, 17, 86, 87, 88, 89, 90, 91, 92, 103, 104]
index = ["TestAnimal"+str(item) for item in animals]
columns = ["Accuracy", "First Hour", "Second Hour", "Third Hour", "First Hour TP", "Second Hour TP",
           "Third Hour TP", "Temporal TP First Hour(%)", "Temporal TP Second Hour(%)", "Temporal TP Third Hour(%)"]
df = pd.DataFrame(columns=columns, index=index)

for ind in index:
    with open(ind+'.json') as file:
        data = json.load(file)
    df["Accuracy"][ind] = round(data["Accuracy"], 3)
    df["First Hour"][ind] = round(data["GroundTruth_in_Minutes"][0], 3)
    df["Second Hour"][ind] = round(data["GroundTruth_in_Minutes"][1], 3)
    df["Third Hour"][ind] = round(data["GroundTruth_in_Minutes"][2], 3)
    df["First Hour TP"][ind] = round(data["True_Positive_in_Minutes"][0], 3)
    df["Second Hour TP"][ind] = round(data["True_Positive_in_Minutes"][1], 3)
    df["Third Hour TP"][ind] = round(data["True_Positive_in_Minutes"][2], 3)
    df["Temporal TP First Hour(%)"][ind] = round(data["Percentages"][0], 3)
    df["Temporal TP Second Hour(%)"][ind] = round(data["Percentages"][1], 3)
    df["Temporal TP Third Hour(%)"][ind] = round(data["Percentages"][2], 3)

NaN = df["Temporal TP Second Hour(%)"]["TestAnimal103"]
df = df.replace(NaN, 1)
first_half = df.iloc[:, 0:7]
second_half = df.iloc[:, 7:]
second_half.style.applymap(lambda item: "background-color: red" if item < 0.70 else (
    "background-color: yellow" if item < 0.90 else "background-color: green"))
df = pd.concat([first_half, second_half], axis=1)
second_half.style.applymap(lambda item: "background-color: red" if item < 0.70 else (
    "background-color: yellow" if item < 0.90 else "background-color: green"))
df.iloc[:, 7:].style.applymap(lambda item: "background-color: red" if item < 0.70 else (
    "background-color: yellow" if item < 0.90 else "background-color: green"))

# %%
df.to_latex("AllRatsResults.tex")
# %%
