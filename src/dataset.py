import pandas as pd
import numpy as np
import random
from labeling import Label
from tqdm import tqdm
from utils import join


def merge(text, tags):
    text_joined = []
    tags_joined = []

    for p,q in zip(text, tags):
        text_joined.extend(p)
        tags_joined.extend(q)

    return text_joined, tags_joined


def preprocess_data():
    data = pd.read_csv('../ner_dataset.csv', encoding='latin-1')
    data['Sentence #'] = data['Sentence #'].fillna(method='ffill')

    data['Tag'].replace(['I-art','B-art'], 'art', inplace=True)
    data['Tag'].replace(['I-eve','B-eve'], 'eve', inplace=True)
    data['Tag'].replace(['I-geo','B-geo'], 'geo', inplace=True)
    data['Tag'].replace(['I-gpe','B-gpe'], 'gpe', inplace=True)
    data['Tag'].replace(['I-nat','B-nat'], 'nat', inplace=True)
    data['Tag'].replace(['I-tim','B-tim'], 'tim', inplace=True)
    data['Tag'].replace(['I-org','B-org'], 'org', inplace=True)
    data['Tag'].replace(['I-per','B-per'], 'per', inplace=True)

    g = data.groupby('Sentence #').agg(lambda x: list(x))

    text = list(g['Word'])
    tags = list(g['Tag'])
    text_joined, tags_joined = join(text, tags)
    text_joined, tags_joined = merge(text_joined, tags_joined)
    df = pd.DataFrame({'Text':text_joined, 'Tags':tags_joined})

    geo = list(df['Text'][df['Tags']=='geo'])
    eve = list(df['Text'][df['Tags']=='eve'])
    art = list(df['Text'][df['Tags']=='art'])
    gpe = list(df['Text'][df['Tags']=='gpe'])
    nat = list(df['Text'][df['Tags']=='nat'])
    tim = list(df['Text'][df['Tags']=='tim'])
    org = list(df['Text'][df['Tags']=='org'])
    per = list(df['Text'][df['Tags']=='per'])

    geo_sample = random.sample(geo, 30)
    eve_sample = random.sample(eve, 30)
    art_sample = random.sample(art, 30)
    gpe_sample = random.sample(gpe, 30)
    nat_sample = random.sample(nat, 30)
    tim_sample = random.sample(tim, 30)
    org_sample = random.sample(org, 30)
    per_sample = random.sample(per, 30)

    samples = [geo_sample, eve_sample, art_sample, gpe_sample, nat_sample, tim_sample, org_sample, per_sample]
    names = ['gpe', 'eve', 'art', 'geo', 'nat', 'tim', 'org', 'per']
    entity_dict = {}
    for sample, name in zip(samples, names):
        for element in sample:
            entity_dict[element] = name

    return text, tags, entity_dict