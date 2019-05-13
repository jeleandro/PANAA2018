import os;
import re;
import json;
import glob;
import codecs;
import zipfile;

from os.path import join as pathjoin;



def readImpostorSample(baseDir, name, language, sampleSize, random_state=None):
    from sklearn.utils.validation import check_random_state
    sample = [];
    
    with zipfile.ZipFile(baseDir+'/'+name+'.zip') as zf:
        for t in zf.namelist():
            if not t.lower().endswith('.txt'):
                continue;
            _id = t.replace(name+'/','');
            if '/' not in _id:
                continue;

            lang, _id = _id.strip().split('/');
            
            if lang != language:
                continue;

            sample.append(t)
            
        random_state = check_random_state(random_state)
        sampleId = random_state.randint(0,len(sample),sampleSize);
    
        sample = [sample[i] for i in sampleId ]
        sample = [ (zf.open(s).read().decode('utf-8'), '<UNK>',None) for s in sample]
    
    return sample;



def readCollectionsOfProblems2019(path):
    import glob, codecs
    dataset = [];
    # Reading information about the collection
    infocollection = path+os.sep+'collection-info.json'
    problems = []
    with open(infocollection, 'r') as f:
        problems = json.load(f);
    for index,problem in enumerate(problems):
        # Reading information about the problem
        basePath = path+os.sep+problem['problem-name'];
        infoproblem = basePath+os.sep+'problem-info.json'
        candidates = []
        with open(infoproblem, 'r') as f:
            fj = json.load(f)
            unk_folder = fj['unknown-folder']
            for attrib in fj['candidate-authors']:
                candidates.append(attrib['author-name'])
        # Building training set
        
        train_set = []
        for candidate in candidates:
            candidate_set = glob.glob(basePath+os.sep+candidate+os.sep+'*.txt');
            candidate_set = [
                [codecs.open(v,'r',encoding='utf-8').read(),candidate, v.split(os.sep)[-1]] for v in candidate_set
            ];
            train_set += candidate_set;
            
        # Building test set
        with open(basePath+os.sep+'ground-truth.json', 'r') as f:
            gt = json.load(f)['ground_truth'];
            gt = {g['unknown-text']:g['true-author'] for g in gt }
        
        test_set = glob.glob(basePath+os.sep+unk_folder+os.sep+'*.txt');
        test_set = [
            [codecs.open(v,'r',encoding='utf-8').read(),gt[v.split(os.sep)[-1]],v.split(os.sep)[-1]] for v in test_set
        ];
        
        problem['candidates'] = train_set;
        problem['unknown'] = test_set;
    return problems;



def readCollectionsOfProblems(path):
    # Reading information about the collection
    infocollection = path+os.sep+'collection-info.json'
    with open(infocollection, 'r') as f:
        problems  = [
            {
                'problem': attrib['problem-name'],
                'language': attrib['language'],
                'encoding': attrib['encoding'],
            }
            for attrib in json.load(f)
            
        ]

    for index,problem in enumerate(problems):
        unk_folder, candidates_folder = readProblem(path, problem['problem']); 
        problem['candidates_folder_count'] = len(candidates_folder);
        problem['candidates'] = [];
        for candidate in candidates_folder:
            problem['candidates'].extend(read_files(pathjoin(path, problem['problem']),candidate));
        
        problem['unknown'] = read_files(pathjoin(path, problem['problem']),unk_folder);    

    return problems;


def readProblem(path, problem):
    # Reading information about the problem
    infoproblem = path+os.sep+problem+os.sep+'problem-info.json'
    candidates = []
    with open(infoproblem, 'r') as f:
        fj = json.load(f)
        unk_folder = fj['unknown-folder']
        for attrib in fj['candidate-authors']:
            candidates.append(attrib['author-name'])
    return unk_folder, candidates;


def read_files(path,label):
    # Reads all text files located in the 'path' and assigns them to 'label' class
    files = glob.glob(pathjoin(path,label,'*.txt'))
    texts=[]
    for i,v in enumerate(files):
        f=codecs.open(v,'r',encoding='utf-8')
        texts.append([f.read(),label, os.path.basename(v)])
        f.close()
    return texts


#*******************************************************************************************************
import warnings
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder


def eval_measures(gt, pred):
    """Compute macro-averaged F1-scores, macro-averaged precision, 
    macro-averaged recall, and micro-averaged accuracy according the ad hoc
    rules discussed at the top of this file.
    Parameters
    ----------
    gt : dict
        Ground truth, where keys indicate text file names
        (e.g. `unknown00002.txt`), and values represent
        author labels (e.g. `candidate00003`)
    pred : dict
        Predicted attribution, where keys indicate text file names
        (e.g. `unknown00002.txt`), and values represent
        author labels (e.g. `candidate00003`)
    Returns
    -------
    f1 : float
        Macro-averaged F1-score
    precision : float
        Macro-averaged precision
    recall : float
        Macro-averaged recall
    accuracy : float
        Micro-averaged F1-score
    """

    actual_authors = list(gt.values())
    encoder = LabelEncoder().fit(['<UNK>'] + actual_authors)

    text_ids, gold_authors, silver_authors = [], [], []
    for text_id in sorted(gt):
        text_ids.append(text_id)
        gold_authors.append(gt[text_id])
        try:
            silver_authors.append(pred[text_id])
        except KeyError:
            # missing attributions get <UNK>:
            silver_authors.append('<UNK>')

    assert len(text_ids) == len(gold_authors)
    assert len(text_ids) == len(silver_authors)

    # replace non-existent silver authors with '<UNK>':
    silver_authors = [a if a in encoder.classes_ else '<UNK>' 
                      for a in silver_authors]

    gold_author_ints   = encoder.transform(gold_authors)
    silver_author_ints = encoder.transform(silver_authors)

    # get F1 for individual classes (and suppress warnings):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f1 = f1_score(gold_author_ints,
                  silver_author_ints,
                  labels=list(set(gold_author_ints)),
                  average='macro')
        precision = precision_score(gold_author_ints,
                  silver_author_ints,
                  labels=list(set(gold_author_ints)),
                  average='macro')
        recall = recall_score(gold_author_ints,
                  silver_author_ints,
                  labels=list(set(gold_author_ints)),
                  average='macro')
        accuracy = accuracy_score(gold_author_ints,
                  silver_author_ints)

    return f1,precision,recall,accuracy


def readGroundTruh(ground_truth_file, unkowndocs):
    gt = {}
    with open(ground_truth_file, 'r') as f:
        for attrib in json.load(f)['ground_truth']:
            gt[attrib['unknown-text']] = attrib['true-author'];

    return [gt[d]  for d in unkowndocs];



def evaluate(ground_truth_file,predictions_file):
    # Calculates evaluation measures for a single attribution problem
    gt = {}
    with open(ground_truth_file, 'r') as f:
        for attrib in json.load(f)['ground_truth']:
            gt[attrib['unknown-text']] = attrib['true-author']

    pred = {}
    with open(predictions_file, 'r') as f:
        for attrib in json.load(f):
            if attrib['unknown-text'] not in pred:
                pred[attrib['unknown-text']] = attrib['predicted-author']
    f1,precision,recall,accuracy =  eval_measures(gt,pred)
    return f1, precision, recall, accuracy