import pandas as pd
from scipy.io.arff import loadarff

DATASETS = {'atp1d.arff': 6, 'atp7d.arff': 6,
            'oes97.arff': 16, 'oes10.arff': 16,
            'rf1.arff': 8, 'rf2.arff': 8,
            'scm1d.arff':16, 'scm20d.arff': 16,
            'edm.arff': 2,
            'sf1.arff': 3, 'sf2.arff': 3,
            'jura.arff': 3,
            'wq.arff': 14,
            'enb.arff': 2,
            'slump.arff': 3,
            'andro.arff': 6,
            'osales.arff': 12,
            'scfp.arff': 3
           }


B_DIR = './data/mtr-datasets/'
def load_arff_dataset(arff_path, n_targets):
    data, _ = loadarff(arff_path)
    df = pd.DataFrame(data=data)
    X = df[df.columns[:-n_targets]].copy()
    y = df[df.columns[-n_targets:]].copy()
    
    return X, y

def load():
    datasets = {}
    for arff_file, n_targets in DATASETS.items():
        datasets[arff_file.split('.')[0]] = load_arff_dataset(B_DIR + arff_file, n_targets)
    return datasets
    