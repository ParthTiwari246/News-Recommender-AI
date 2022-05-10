import pandas as pd
import numpy as np

newsdatabase_df = pd.read_csv("CSVs/newsdatabase.csv")
newsdatabase_df.drop('Unnamed: 0', axis=1, inplace=True)

business_df = newsdatabase_df[newsdatabase_df['type']=='Business']
business_df.to_csv('CSVs/business.csv')



entertainment_df = newsdatabase_df[newsdatabase_df['type']=='Entertainment']
entertainment_df.to_csv('CSVs/entertainment.csv')


tech_df = newsdatabase_df[newsdatabase_df['type']=='Tech']
tech_df.to_csv('CSVs/tech.csv')

politics_df = newsdatabase_df[newsdatabase_df['type']=='Politics']
politics_df.to_csv('CSVs/politics.csv')


sport_df = newsdatabase_df[newsdatabase_df['type']=='Sports']
sport_df.to_csv('CSVs/sport.csv')