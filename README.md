Assignment for CIT 42100/52600 - Merge

1. Given the Data Set (adult-data-new.csv) and its description as follows: Please use k-means clustering algorithms to see if you can find the clusters of persons who make over 50K or below 50K. 
Please include the screenshots of the results from Weka or Python code in your report. 
NOTE: You will need to do some data preprocessing steps, such as data encoder and data normalization. 

2. Given the bakery data set (Bakery.csv). please use association rule mining to identify the most valuable frequent item sets/item association rules that can help the business to create better business plans.  

Please explain the criteria that are used to choose the valuable rules:
# Criteria for choosing valuable rules:
# 1. High support: Rules with high support values (>0.05) are more likely to occur frequently in real transactions.
# 2. High confidence: Rules with high confidence values (>0.7) indicate strong associations between items.
# 3. High lift: Rules with high lift values (>1.5) suggest stronger relationships than random chance.
# 4. Relevance to business goals: We'll focus on rules involving popular items or those that could improve sales.

This data set is built from bakery chain has a menu of about 40 pastry items and 10 coffee drinks. It has a number of locations in West Coast states (California, Oregon, Arizona, Nevada).  
If you use python code, please make sure you convert the data to the format that python program takes

Data set description:
The Bakery.csv file format is: receipt# followed by No's and Yes's indicating if an item was on a given receipt. (full binary vector representation) 