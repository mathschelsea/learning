import pandas as pd
import yaml
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def main():

    # importing the main dataset
    logging.info('Importing the Bluebook for Bulldozers Dataset')
    df = pd.read_csv('data/bbfb/Train.csv', low_memory=False, parse_dates=['saledate'])
    print(f'No. rows: {df.shape[0]}')
    print(f'No. cols: {df.shape[1]}')

    # importing feature spec
    with open('meta/feature_spec.yaml', 'r') as feature_spec:
    feat_dict = yaml.safe_load(feature_spec)

    # performing data corrections
    logging.info('Performing Data Corrections')
    df.Tire_Size = df.Tire_Size.str.replace('"','')
    df.Tire_Size = df.Tire_Size.str.replace(' inch','') 
    df.Tire_Size.fillna('0', inplace=True)
    df.Tire_Size = df.Tire_Size.str.replace('None or Unspecified', '-1')
    df.Tire_Size = df.Tire_Size.astype(float)

    df.Undercarriage_Pad_Width = df.Undercarriage_Pad_Width.str.replace('.5','')
    df.Undercarriage_Pad_Width = df.Undercarriage_Pad_Width.str.replace(' inch','')
    df.Undercarriage_Pad_Width.fillna('0', inplace=True)
    df.Undercarriage_Pad_Width = df.Undercarriage_Pad_Width.str.replace('None or Unspecified', '-1')
    df.Undercarriage_Pad_Width = df.Undercarriage_Pad_Width.astype(int)

    df.Blade_Width = df.Blade_Width.str.replace("'","")
    df.Blade_Width = df.Blade_Width.str.replace("<12","11")
    df.Blade_Width.fillna('0', inplace=True)
    df.Blade_Width = df.Blade_Width.str.replace('None or Unspecified', '-1')
    df.Blade_Width = df.Blade_Width.astype(int)

    df.Transmission = df.Transmission.str.replace('AutoShift', 'Autoshift')

    # grouping categorical variables
    logging.info('Grouping Categorical Variables')
    threshold = 50
    for c in ['fiSecondaryDesc', 'fiModelSeries', 'fiModelDescriptor']:
    counts = df[c].value_counts(dropna=False)
    flag = df[c].isin(counts.index[counts < threshold])
    df.loc[flag, c] = 'Grouped'

if __name__ == "__main__":
    main()