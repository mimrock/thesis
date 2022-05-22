import argparse
import pandas as pd
from pandas_profiling import ProfileReport

parser = argparse.ArgumentParser()

parser.add_argument('--data',
                    help='Path to the csv file containing the data.', required=True)

parser.add_argument('--output',
                    help='Path to the html file to be generated.', required=True)
args = parser.parse_args()

df = pd.read_csv(args.data)
pd.options.display.max_columns = 10
print(df.describe())

profile = ProfileReport(df, title="Pandas Profiling Report")
profile.to_file(args.output)