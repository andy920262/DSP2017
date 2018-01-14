import numpy as np
itop = ['','0','1','2','3','4','5','6','7','8','9','','']
ptoi = {'.':0,'liN':1,'#i':2,'#er':3,'san':4,'sy':5,'#u':6,'liou':7,'qi':8,'ba':9,'jiou':10,'sil':11,'sp':12,
        'ling':1,'yi':2,'er':3,'si':5,'wu':6,'liu':7,'ba':9,'jiu':10}

def load_data(feat_path, label_path):
    
    feat_file = open(feat_path, 'r')
    id, feat = [], []
    for line in feat_file:
        line = line.strip()
        if line[0] == 'L':
            id.append(line.split(' ')[-1].split('/')[-1].split('.')[0])
            feat.append([])
        else:
            feat[-1].append(line.split(' '))
    for f in feat:
        f += [[0.0] * 39] * (302 - len(f))
    feat = np.array(feat).astype(np.float64)
    
    label_file = open(label_path, 'r')
    label_file.readline()
    label = []
    used = False
    for line in label_file:
        line = line.strip()
        if line[0] == '"':
            used = line.split('.')[0].split('/')[1] in id
            if used:
                label.append([])
        else:
            if used:
                label[-1].append(ptoi[line])
    for l in label:
        l += [0] * (16 - len(l))
    label = np.array(label).astype(np.int64)
    return id, feat, label

if __name__ == '__main__':
    id, feat, label = load_data('data/train.dat', 'labels/Clean08TR.mlf')
