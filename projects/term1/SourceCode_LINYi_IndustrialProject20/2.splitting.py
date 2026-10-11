import pandas as pd

df = pd.read_csv("guardian_news_ai_labeled.csv")

# Downstream graph construction expects only usable LLM label rows. Rows with
# ai_status == "error" are kept in the labeled file for auditing, but excluded
# from train/val/test to avoid treating API failures as true negative labels.
if "ai_status" in df.columns:
    df = df[df["ai_status"].ne("error")].copy()

df["webPublicationDate"] = pd.to_datetime(df["webPublicationDate"])
df = df.sort_values("webPublicationDate").reset_index(drop=True)

n = len(df)
n_train = int(n * 0.6)
n_val = int(n * 0.2)

train_df = df.iloc[:n_train].copy()
val_df   = df.iloc[n_train:n_train + n_val].copy()
test_df  = df.iloc[n_train + n_val:].copy()

train_df.to_csv("guardian_news_train.csv", index=False)
val_df.to_csv("guardian_news_val.csv", index=False)
test_df.to_csv("guardian_news_test.csv", index=False)

print(f"Saved guardian_news_train.csv rows={len(train_df)}")
print(f"Saved guardian_news_val.csv rows={len(val_df)}")
print(f"Saved guardian_news_test.csv rows={len(test_df)}")
