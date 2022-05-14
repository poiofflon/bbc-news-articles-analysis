import pandas as pd
import config
from spacy.cli import download


def download_spacy_data():
    download("en_core_web_sm")


def read_bbc_csv():
    df = pd.read_csv(config.bbc_data)
    df = remove_duplicates(df)
    validate_data(df)
    return df


def validate_data(df):
    # check is as expected and described
    for category in df.category.unique():
        assert category in ['business', 'entertainment', 'politics', 'sport', 'tech']
    assert not any(df.duplicated())


def remove_duplicates(df):
    if any(df.duplicated()):
        dup = df.duplicated().value_counts()
        df = df[~df.duplicated()]
        print(f'{dup[True]} duplicated articles removed from dataset')
    return df




