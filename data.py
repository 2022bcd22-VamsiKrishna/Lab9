import pandas as pd

main_df = pd.read_csv('housing.csv')

sub_df = main_df[:5000]

sub_df.to_csv('data/housing.csv', index=False)