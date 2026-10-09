"""Inspect the blank-label rows in the WebAttacks CSV. Read-only."""
import glob
import pandas as pd

web = glob.glob(r"ml\data\raw\cicids2017\*\Thursday-WorkingHours-Morning-WebAttacks*.csv")[0]
inf = glob.glob(r"ml\data\raw\cicids2017\*\Thursday-WorkingHours-Afternoon-Infilteration*.csv")[0]
cols = ["Flow ID", " Timestamp", " Label"]
w = pd.read_csv(web, usecols=cols, dtype=str, keep_default_na=False,
                encoding_errors="replace")
i = pd.read_csv(inf, usecols=cols, dtype=str, keep_default_na=False,
                encoding_errors="replace")

is_blank = w[" Label"].str.strip() == ""
blank = w[is_blank]
print("total rows:", len(w))
print("blank rows:", len(blank))
print("labelled rows:", (~is_blank).sum())
print("blank row index range:", blank.index.min(), "to", blank.index.max())
gaps = blank.index.to_series().diff().dropna()
print("blank rows in one contiguous block:", (gaps == 1).all())

inf_keys = set(zip(i["Flow ID"], i[" Timestamp"]))
blank_keys = list(zip(blank["Flow ID"], blank[" Timestamp"]))
overlap = sum(k in inf_keys for k in blank_keys)
print("blank rows whose (Flow ID, Timestamp) also appear in Infiltration file:", overlap)

print("\nfirst 3 blank rows:")
print(blank.head(3).to_string())