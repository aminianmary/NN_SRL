from dynet import *
import os
from optparse import OptionParser
from train import SRL

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE", default=None)
    parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE", default='')
    parser.add_option("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE", default=None)
    parser.add_option("--output", dest="output", help="output file", metavar="FILE", default=None)
    parser.add_option("--outdir", dest="outdir", help="Output Directory", metavar="FILE")
    parser.add_option("--extern", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--batch", type="int", dest="batch", default=10)
    parser.add_option("--epochs", type="int", dest="epochs", default=30)
    parser.add_option("--min_word_freq", type="int", dest="min_freq", default=1)
    parser.add_option("--k", type="int", dest="k", default=4)
    parser.add_option("--eps", type="float", dest="eps", default=0.00000001)
    parser.add_option("--re_dim", type="int", dest="re_dim", default=100)
    parser.add_option("--le_dim", type="int", dest="le_dim", default=100)
    parser.add_option("--pos_dim", type="int", dest="pos_dim", default=16)
    parser.add_option("--lstm_dim", type="int", dest="lstm_dim", default=512)
    parser.add_option("--r_dim", type="int", dest="r_dim", default=128)
    parser.add_option("--le_prime_dim", type="int", dest="le_prime_dim", default=128)
    parser.add_option("--dynet-mem", type="int", default=10240)

    (options, args) = parser.parse_args()
    if options.conll_train:
        print 'Initializing blstm srl:'
        parser = SRL(options.conll_train, options.external_embedding, options)
        for epoch in xrange(options.epochs):
            print 'Starting epoch', epoch
            parser.train(options.conll_train)
        parser.Save(os.path.join(options.outdir, options.model))




