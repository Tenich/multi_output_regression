from online_sales import load as online_sales_dataset
from arf_datasets import load as arf_datasets

ALL_DATASETS = [online_sales_dataset, arf_datasets]

def get_datasets():
    datasets = {}
    for ds in ALL_DATASETS:
        datasets.update(ds())
    return datasets
    