import codecs

import pandas as pd
import nltk
import operator

tag_set = set()

for f in ["train", "dev", "test"]:

  hyp2label = {}

  lbls, hypoths, premises = [], [], []
  for line in codecs.open("scitail/SciTailV1/tsv_format/scitail_1.0_%s.tsv" % (f), encoding="utf-8"):
    if f == "dev":
      f = "val"

    line = line.strip().split("\t")
    assert(len(line) == 3) 
    tag = line[-1]
    if tag == "entails":
      tag = "entailment"
    tag_set.add(tag)
    lbls.append(tag)
    hyp = line[1]
    hypoths.append(hyp) 
    premises.append(line[0])

    if f == 'train':
      if hyp not in hyp2label:
        hyp2label[hyp] = {}
      if tag not in hyp2label[hyp]:
        hyp2label[hyp][tag] = 0
      hyp2label[hyp][tag] += 1

  lbl_out = codecs.open("scitail/cl_scitail_%s_lbl_file" % (f), "wb", encoding="utf-8")
  source_out = codecs.open("scitail/cl_scitail_%s_source_file" % (f), "wb", encoding="utf-8")
  for i in range(len(lbls)):
    lbl_out.write(lbls[i].strip() + "\n")
    source_out.write(" ".join(nltk.word_tokenize(premises[i])) + "|||" + " ".join(nltk.word_tokenize(hypoths[i].strip())) + "\n")

  lbl_out.close()
  source_out.close()

