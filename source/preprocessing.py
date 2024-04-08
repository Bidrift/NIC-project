import pandas as pd
import numpy as np

file_path = 'Dataset.xlsx'
data = pd.read_excel(file_path)


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        'Time': 'year',
        'Country Name': 'country_name',
        'Country Code': 'country_code',
        'GDP (current US$) [NY.GDP.MKTP.CD]': 'gdp_current',
        'GDP per capita (current US$) [NY.GDP.PCAP.CD]': 'gdp_per_capita',
        'Population growth (annual %) [SP.POP.GROW]': 'population_growth',
        'Life expectancy at birth, female (years) [SP.DYN.LE00.FE.IN]': 'life_expectancy_female',
        'Life expectancy at birth, male (years) [SP.DYN.LE00.MA.IN]': 'life_expectancy_male',
        'Mortality rate, adult, male (per 1,000 male adults) [SP.DYN.AMRT.MA]': 'mortality_rate_adult_male',
        'Fertility rate, total (births per woman) [SP.DYN.TFRT.IN]': 'fertility_rate',
        'Inflation, consumer prices (annual %) [FP.CPI.TOTL.ZG]': 'inflation_consumer_prices',
        'GDP growth (annual %) [NY.GDP.MKTP.KD.ZG]': 'gdp_growth',
        'GDP per capita growth (annual %) [NY.GDP.PCAP.KD.ZG]': 'gdp_per_capita_growth',
        'Foreign direct investment, net (BoP, current US$) [BN.KLT.DINV.CD]': 'foreign_direct_investment',
        'Net migration [SM.POP.NETM]': 'net_migration',
        'Mortality rate, adult, female (per 1,000 female adults) [SP.DYN.AMRT.FE]': 'mortality_rate_adult_female',
        'Death rate, crude (per 1,000 people) [SP.DYN.CDRT.IN]': 'death_rate',
        'Rural population [SP.RUR.TOTL]': 'rural_population',
        'Age dependency ratio, young (% of working-age population) [SP.POP.DPND.YG]': 'age_dependency_ratio_young',
        'Age dependency ratio, old (% of working-age population) [SP.POP.DPND.OL]': 'age_dependency_ratio_old'
    })
    df['year'] = df['year'].astype(pd.Int64Dtype())

    # Convert the country's name and code to lower case for uniformity
    df['country_name'] = df['country_name'].astype(str).apply(lambda x: x.lower().strip())
    df['country_code'] = df['country_code'].astype(str).apply(lambda x: x.lower().strip())

    # Identify numeric columns for conversion
    numeric_columns = [col for col in df.columns if col not in ['country_name', 'country_code', 'year']]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    return df


# Preprocess the dataset
data = preprocess_df(data)


# Function to interpolate missing numeric values in the dataset
def interpolate_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('year')

    # Interpolate missing values for numeric columns
    for col in numeric_cols:
        # Attempt linear interpolation first
        df[col] = df[col].interpolate(method='linear')

        # For any remaining missing values, fill forwards first, then backwards
        df[col] = df[col].ffill()
        df[col] = df[col].bfill()
    return df


data = interpolate_missing_values(data)

processed_file_path = 'Dataset_Processed.xlsx'
data.to_excel(processed_file_path, index=False)

print(f"Processed dataset saved to: {processed_file_path}")
