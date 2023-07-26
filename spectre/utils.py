import numpy as np
from time import time
from datetime import datetime
import json, os, lzma, sys, csv
import dill as pickle


def init_qrh(parameters):
    sys.path.insert(1, parameters['quantarhei_dir'])
    global qr
    import quantarhei as qr
    
### https://en.wikipedia.org/wiki/ANSI_escape_code#:~:text=SGR%20(Select%20Graphic%20Rendition)%20parameters
def bold_text(string):
    return "\033[1m{}\033[0m".format(string)

def green_bckg(string):
    return "\033[48;2;204;247;203m{}\033[0m".format(string)

def red_bckg(string):
    return "\033[48;2;247;206;203m{}\033[0m".format(string)


def f_time(t):
    if t < 0.001:
        return "{:.0f}us".format(t*10e5)
    elif t < 1.0:
        return "{:.0f}ms".format(t*10e2)
    elif t < 60:
        return "{:.2f}s".format(t)
    elif t < 3600:
        return "{:.0f}min {:.0f}s".format(t/60,t%60)
    else:
        return "{:.0f}h {:.0f}min {:.0f}s".format(t/3600, (t%3600)/60, t%60)

    
def print_times(labels):
    for label, time in labels.items():
        print(bold_text(label + ": ") + "{}".format(f_time(time)))

        
def check_path(path):
    number = 0
    
    while os.path.isfile(path) is True:
        pth, ext = path.rsplit('.', 1)
        
        if number == 0:
            path = "_{:03d}.".format(number).join([pth, ext])
        else:
            path = "_{:03d}.".format(number).join([pth[:-4], ext]) 
        number += 1
    
    path = path.replace('\\','/')
    dirs = path.rsplit('/', 1)[0]
    
    if not os.path.isdir(dirs):
        os.mkdir(dirs)

    return path


def save2file(obj, path):
    with lzma.open(path, "wb") as f:
        pickle.dump(obj, f)

def loadfile(path):
    with lzma.open(path, "r") as f:
        data = pickle.load(f)
    return data
            
def save2json(dct, path, indent=4):
    with open(path, "w") as f:
        js = json.dumps(dct, indent=indent)
        f.write(js)

def loadjson(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data
        
def load_binary(path=None, home_path="C:/Users/micha/Documents/Studium/MScThesis/data-2/", 
                  save_dir=None, file=None):
    if (home_path is not None) and (save_dir is not None) and (file is not None):
        fpath = os.path.join(home_path, save_dir, file)
    else:
        fpath = path
    assert fpath is not None
    
    with lzma.open(fpath, 'rb') as f:
        data = pickle.load(f)
    return data


def load_csv(path, delimiter=';', decimal=','):
    dataset = list()
    with open(path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=delimiter)
        for row in csv_reader:
            dataset.append([s.replace(decimal, '.') for s in row])
    
    newx = list()
    newy = list()
    for x, y in dataset:
        if x not in newx:
            newx.append(x)
            newy.append(y)
        else:
            continue
            
    dataset = np.array([newx,newy], dtype=float)
    ordered = np.argsort(dataset)[0]
    out = dataset[:,ordered]
    
    return out
    
            
def get_differences_in_lists(*lists):
    k = len(lists[0])
    
    for i in range(k):
        diffs = [lists[0][i], ]
        for l in lists[1:]:
            if l[i] != lists[0][i]:
                diffs.append(l[i])
            else:
                break
        if len(diffs) > 1:
            return i, diffs

def get_differences_in_dicts(*dicts, omit=['timedate','file','save_dir'], use_as_labels=None, key0=''):
    if (use_as_labels is not None) and (use_as_labels in dicts[0].keys()):
        return use_as_labels, [d[use_as_labels] for d in dicts]
    
    keys = list()
    
    for d in dicts[1:]:
        keys = [k for k in d.keys() if k in dicts[0] and d[k] != dicts[0][k] and k not in omit]

    #print(keys)
    keys = [list(dict.fromkeys(keys))[-1]]
    #print(keys)
    #assert len(keys) == 1, "Too many (or none) differences found in the parameter files: {}".format(len(keys))
    diffs = [d[keys[0]] for d in dicts]
    
    key0 += '{}'.format(keys[0])

    if type(diffs[0]) == dict:
        k, diffs = get_differences_in_dicts(*diffs, omit=omit, use_as_labels=use_as_labels)
        
    elif type(diffs[0]) == list:
        k, diffs = get_differences_in_lists(*diffs)

    else:
        return key0, diffs

    key0 += ':{}'.format(k)
    return key0, diffs

def angle_to_vects(angle, dy, dx, vecx=[1,0,0]):
    vecx = np.array([1,0,0]) * dx
    vecy = [np.cos(angle)*dy,np.sin(angle)*dy,0]
    pp = dict()
    pp['dipx'] = list(vecx)
    pp['dipy'] = list(vecy)
    pp['angle'] = angle
    return pp
