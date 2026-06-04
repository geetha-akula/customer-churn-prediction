import pandas as pd 

df = pd.read_excel("Data/Telco_customer_churn_dataset.xlsx")
df.to_csv("Data/telco_churn.csv", index= False)