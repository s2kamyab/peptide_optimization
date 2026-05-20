import wandb
import os
api = wandb.Api()

# fill these in from your run page URL
# fill these in from your run page URL
entity  = "shima-kamyab-university-of-victoria"
project = "ankibind-lolbo-peptides"  # or whatever you used
run_id  = "20260224-020050"                  # the run id (last part of the run url)

run = api.run(f"{entity}/{project}/{run_id}")

df = run.history(samples=20000)   # full history (increase if needed)
df.to_csv("wandb_history.csv", index=False)

summary = dict(run.summary)
print("summary keys:", list(summary.keys())[:50])