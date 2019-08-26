import pandas as pd
import nltk
import json

for f in ["train", "dev", "test", "test_hard"]:
  if f == "test_hard":
    with open("snli_1.0/snli_1.0_%s.jsonl" % (f)) as f_hard:
      sentence_ones, sentence_twos, labels = [], [], []
      for line in f_hard:
        d = json.loads(line.strip())
        sentence_ones.append(d['sentence1'])
        sentence_twos.append(d['sentence2'])
        labels.append(d['gold_label'])

  else:
    df = pd.read_table("snli_1.0/snli_1.0_%s.txt" % (f))
    if f == "dev":
      f = "val"

    sentence_ones = df['sentence1']
    sentence_twos = df['sentence2']
    labels = df['gold_label']

  assert(len(labels) == len(sentence_ones) == len(sentence_twos))
  lbl_out = open("snli_1.0/cl_snli_%s_lbl_file" % (f), "wb")
  source_out = open("snli_1.0/cl_snli_%s_source_file" % (f), "wb")
  label_set = set(["entailment","neutral","contradiction"])
  for i in range(len(labels)):
    if labels[i] not in label_set:
      continue
    try:
      if sentence_twos[i].isdigit() or sentence_ones[i].isdigit():
        continue
      lbl_out.write(labels[i].strip() + "\n")
      source_out.write(" ".join(nltk.word_tokenize(sentence_ones[i].strip())) + "|||" + " ".join(nltk.word_tokenize(sentence_twos[i].strip())) + "\n")
    except:
      continue
      # There are a lot of examples where only the premise sentence was given
      # The sentence is often something like: cannot see the picture

  lbl_out.close()
  source_out.close()
