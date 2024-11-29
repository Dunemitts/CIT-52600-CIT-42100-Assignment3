import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#load the data
df = pd.read_csv(r'data\Bakery.csv', index_col=0)

#converts all yes and no values to boolean outcomes
for col in df.columns[1:]:
    df[col] = df[col].replace({'Yes': True, 'No': False})

#convert the data to a list of transactions
transactions = df.iloc[:, 1:].values.tolist()

#create and encode transactions
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
data_df = pd.DataFrame(te_ary, columns=te.columns_)

#check for null values
print(f"check nulls: \n{data_df.isnull().sum()}")
print(f"check frequency: \n{data_df.value_counts()}") 

#find frequent itemsets
frequent_itemsets = apriori(data_df, min_support=0.001, use_colnames=True)
print(f"Frequent_Itemsets: \n{frequent_itemsets}")

#generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7, num_itemsets=len(frequent_itemsets))

#sort rules by lift in descending order
rules = rules.sort_values('lift', ascending=False)

#filter rules based on these criteria
valuable_rules = rules[(rules['support'] > 0.01) & (rules['confidence'] > 0.5) & (rules['lift'] > 1.0)]

#calculate item popularity
item_popularity = df.iloc[:, 1:].mean().sort_values(ascending=False)

#visualize the rules
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.bar(item_popularity.index, item_popularity.values)
plt.title('Popularity of Items')
plt.xlabel('Item Name')
plt.ylabel('Popularity Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"Valuable Rules: \n{valuable_rules.head(10)}")
print(f"Popular Items: \n{item_popularity.head(10)}")
