from dynet import *
from util import *
import numpy as np
import random, time

class SRL:
    def __init__(self, train_fp, exter_embed_fp, options):
        self.train_file_path = train_fp
        self.exter_embedding_path = exter_embed_fp
        self.min_freq = options.min_freq
        self.word_freq_dic, self.lemma_freq_dic, self.w2i, self.pos2i, self.r2i, self.lem2i = vocab(train_fp, options.min_freq)
        self.exter_embed, self.exter_embed_w2i = read_external_embedding(exter_embed_fp)
        self.model = Model()
        self.epochs = options.epochs
        self.NUM_LAYERS = options.k
        self.x_re_embed_dim = options.re_dim
        self.x_pe_embed_dim = len(self.exter_embed.values()[0])
        self.x_pos_embed_dim = options.pos_dim
        self.x_le_embed_dim = options.le_dim
        self.x_le_prime_dim = options.le_prime_dim
        self.r_dim = options.r_dim
        self.lstm_dim = options.lstm_dim
        self.batch = options.batch  # batch size

        self.x_embed_dim = self.x_re_embed_dim + self.x_pe_embed_dim + self.x_pos_embed_dim + self.x_le_embed_dim + 1
        self.null_role = '_'

        self.lstm = BiRNNBuilder(self.NUM_LAYERS, self.x_embed_dim, self.lstm_dim, self.model, LSTMBuilder)
        self.x_pe = self.model.add_lookup_parameters((len(self.exter_embed) + 1, self.x_pe_embed_dim))
        self.x_re = self.model.add_lookup_parameters((len(self.w2i) + 1, self.x_re_embed_dim))
        self.x_pos = self.model.add_lookup_parameters((len(self.pos2i) + 1, self.x_pos_embed_dim))
        self.x_le = self.model.add_lookup_parameters((len(self.lem2i) + 1, self.x_le_embed_dim))
        self.u_l = self.model.add_lookup_parameters((len(self.lem2i) + 1, self.x_le_prime_dim))
        self.v_r = self.model.add_lookup_parameters((len(self.r2i), self.r_dim))
        self.U = self.model.add_parameters((self.lstm_dim * 2, self.r_dim + self.x_le_prime_dim))
        self.trainer = AdamTrainer(self.model)

        for word in self.exter_embed_w2i:
            word_index = self.exter_embed_w2i[word]
            self.x_pe.init_row(word_index + 1, self.exter_embed[word])
        self.x_pe.init_row(0, [0.0 for _ in xrange(self.x_pe_embed_dim)])  # UNK embedding
        self.x_pe.set_updated(False)
        self.x_re.init_row(0, [0.0 for _ in xrange(self.x_re_embed_dim)])
        print 'Load external embedding. Vector dimensions', self.x_pe_embed_dim

    def getBiLSTMFeatures (self, sentence):
        lstm_output = {}
        words = sentence.words
        pos_tags = sentence.pos_tags
        lemmas = sentence.lemmas
        preds = sentence.preds

        wpembed, wrembed, posembed = [], [], []
        for j in xrange(len(words)):
            wpembed.append(self.x_pe[self.exter_embed_w2i.get(words[j]) + 1] if words[j] in self.exter_embed_w2i else self.x_pe[0])
            wrembed.append(self.x_re[self.w2i.get(words[j]) + 1] if words[j] in self.w2i else self.x_re[0])
            posembed.append(lookup(self.x_pos, self.pos2i.get(pos_tags[j])))

        for i in xrange(len(preds)):
            pred_index = preds[i][0]

            lemembed, predflag = [], []
            for k in xrange(len(words)):
                lemembed.append(lookup(self.x_le, self.lem2i.get(lemmas[k]))) if j == pred_index else lemembed.append(
                    inputVector([0] * self.x_le_embed_dim))
                predflag.append(inputVector([1])) if k == pred_index else predflag.append(inputVector([0]))

            x_embed = [
                concatenate(filter(None, [wpembed[i], wrembed[i], posembed[i], lemembed[i], predflag[i]])) for i in
                xrange(len(wpembed))]
            lstm_output[pred_index] = self.lstm.transduce(x_embed)  # bilstm features for this sentence-predicate
        return lstm_output

    def buildGraph(self, sentence):
        loss_values = []
        words = sentence.words
        lemmas = sentence.lemmas
        preds = sentence.preds
        args = sentence.args

        for i  in xrange(len(preds)):
            pred_index = preds[i][0]
            lstm_output = self.getBiLSTMFeatures(sentence)[pred_index]
            rind = {windex: self.r2i[role] for windex, role in args[i]}

            W = transpose(concatenate_cols(
                [rectify(self.U.expr() * (concatenate([self.u_l[self.lem2i[lemmas[pred_index]]], self.v_r[role]]))) for role in
                 xrange(len(self.r2i))]))

            for windex in xrange(len(words)):
                gold_role = rind[windex] if windex in rind else self.r2i[self.null_role]
                v_i = lstm_output[windex]
                v_p = lstm_output[pred_index]
                scores = W * concatenate([v_i, v_p])
                loss_values.append(pickneglogsoftmax(scores, gold_role))
        return loss_values

    def train(self, conll_path):
        print 'Training started.'
        start = time.time()
        shuffledData = list(read_conll(conll_path))
        random.shuffle(shuffledData)
        errs, loss, corrects, iters, sen_num = [], 0, 0, 0, 0
        for iSentence, sentence in enumerate(shuffledData):
            e = self.buildGraph(Sentence(sentence))
            errs += e
            sen_num += 1
            if sen_num >= self.batch and len(errs) > 0:
                sum_errs = esum(errs) / len(errs)
                loss += sum_errs.scalar_value()
                sum_errs.backward()
                self.trainer.update()
                renew_cg()
                print 'loss:', loss, 'time:', time.time() - start, 'sen#', (iSentence + 1), 'instances', len(errs)
                errs, loss, sen_num = [], 0, 0
                iters += 1
                start = time.time()
        self.trainer.update_epoch()

    def decode(self, sentence):
        lstm_output = self.getBiLSTMFeatures(sentence)
        preds = sentence.preds
        lemmas = sentence.lemmas

        for i in xrange(len(preds)):
            pred_index = preds[i][0]
            pred_lemma = lemmas[pred_index]
            pred_lemma_index = self.lem2i[pred_lemma] if pred_lemma in self.lem2i else 0
            v_p = lstm_output[pred_index]
            W = transpose(concatenate_cols([rectify(self.U.expr() * (concatenate([self.u_l[pred_lemma_index], self.v_r[role]]))) for role in
                 xrange(len(self.r2i))]))
            for word_index in xrange(len(sentence.words)):
                v_i = lstm_output[pred_index][word_index]
                scores = W * concatenate([v_i, v_p])
                predication = np.argmax(scores.npvalue())
                prediction_index = self.r2i(predication)
                if predication != self.null_role:
                    sentence.args[i] = (word_index, prediction_index)

    def Predict(self, conll_path):
        for iSentence, sentence in enumerate(read_conll(conll_path)):
            self.decode(sentence)
            renew_cg()
            yield sentence

    def Save(self, filename):
        self.model.save(filename)

    def Load(self, filename):
        self.model.load(filename)