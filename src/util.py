import re, codecs
from collections import Counter, defaultdict

class ConllEntry:
    def __init__(self, id, form, lemma, pos, arguments, parent_id=-1, dep='_', is_pred=False, predicate_sense = "_"):
        self.id = id
        self.form = form
        self.lemma = lemma
        self.norm = normalize(form)
        self.lemmaNorm = normalize(lemma)
        self.pos = pos.upper()
        self.parent_id = parent_id
        self.dep = dep
        self.is_pred = is_pred
        self.predicate_sense = predicate_sense
        self.arguments = arguments #a list of arguments

class Sentence:
    def __init__(self, conll_entries):
        words = []
        pos_tags = []
        lemmas = []
        preds = []
        args = {}

        for conll_entry  in conll_entries:
            words.append(conll_entry.norm)
            pos_tags.append(conll_entry.pos)
            lemmas.append(conll_entry.lemmaNorm)
            if conll_entry.is_pred:
                preds.append((conll_entry.id, conll_entry.predicate_sense))

            for i in xrange(len(conll_entry.arguments)):
                pred_order = i
                if not pred_order in args:
                    args[pred_order] = []
                if conll_entry.arguments[i] != '_':
                    args[pred_order].append((conll_entry.id, conll_entry.arguments[i]))

        self.words = words
        self.pos_tags = pos_tags
        self.lemmas = lemmas
        self.preds = preds
        self.args = args

def vocab(conll_path, min_freq):
    wordsCount = Counter() #wordDic
    posCount = Counter() #posDic
    semRoleCount = Counter() #semRoleDix
    lemma_count = Counter() #lemDic

    for sentence in read_conll(conll_path):
        wordsCount.update([entry.norm for entry in sentence])
        posCount.update([entry.pos for entry in sentence])
        for entry in sentence:
            if entry.is_pred:
                lemma_count.update([entry.lemmaNorm])
            for arg in entry.arguments:
                semRoleCount.update([arg])

    w2i = dict()
    r2i = dict()
    c = 0
    for w in wordsCount.keys():
        if wordsCount[w]> min_freq:
            w2i[w]=c
            c+=1
    b=0
    for r in semRoleCount.keys():
        r2i[r] = b
        b+=1

    return (wordsCount, lemma_count, w2i,
            {p: i for i, p in enumerate(posCount)}, r2i,
            {l: i for i, l in enumerate(lemma_count)})

def read_external_embedding(fp):
    external_embedding_fp = open(fp, 'r')
    external_embedding_fp.readline()
    external_embedding = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line in
                          external_embedding_fp}
    external_embedding_fp.close()
    extern_embed_w2i = {w:i for i,w in enumerate(external_embedding)}
    return (external_embedding, extern_embed_w2i)


def read_conll(fp):
    sentences = codecs.open(fp, 'r').read().strip().split('\n\n')
    read = 0
    for sentence in sentences:
        entries = []
        lines = sentence.strip().split('\n')
        for entry in lines:
            spl = entry.split('\t')
            is_pred = False
            if spl[12] == 'Y':
                is_pred = True
            entries.append(ConllEntry(int(spl[0]) - 1, spl[1], spl[3], spl[5], spl[14:], int(spl[9]), spl[11],
                                      is_pred, spl[13]))
        read += 1
        yield entries
    print read, 'sentences read.'

numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");
def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()

