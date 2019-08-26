import os
import sys
import time
import argparse
import pdb

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from data import get_nli_text, build_vocab, get_batch
from models import SharedNLINet, SharedHypothNet, BLSTMEncoder
from mutils import get_optimizer

def get_args():
  parser = argparse.ArgumentParser(description='Training NLI model based on just hypothesis sentence')

  # paths
  parser.add_argument("--embdfile", type=str, default='../data/embds/glove.840B.300d.txt', help="File containin the word embeddings")
  parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
  parser.add_argument("--outputmodelname", type=str, default='model.pickle')
  parser.add_argument("--outputmodelname_adversary", type=str, default='model_adv.pickle')
  parser.add_argument("--train_lbls_file", type=str, default='../data/snli_1.0/cl_snli_train_lbl_file', help="NLI train data labels file")
  parser.add_argument("--train_src_file", type=str, default='../data/snli_1.0/cl_snli_train_source_file', help="NLI train data source file")
  parser.add_argument("--val_lbls_file", type=str, default='../data/snli_1.0/cl_snli_val_lbl_file', help="NLI validation (dev) data labels file")
  parser.add_argument("--val_src_file", type=str, default='../data/snli_1.0/cl_snli_val_source_file', help="NLI validation (dev) data source file")
  parser.add_argument("--test_lbls_file", type=str, default='../data/snli_1.0/cl_snli_test_lbl_file', help="NLI test data labels file")
  parser.add_argument("--test_src_file", type=str, default='../data/snli_1.0/cl_snli_test_source_file', help="NLI test data source file")

  # data
  parser.add_argument("--max_train_sents", type=int, default=10000000, help="Maximum number of training examples")
  parser.add_argument("--max_val_sents", type=int, default=10000000, help="Maximum number of validation/dev examples")
  parser.add_argument("--max_test_sents", type=int, default=10000000, help="Maximum number of test examples")
  parser.add_argument("--remove_dup", type=int, default=0, help="Whether to remove hypotheses that are duplicates. 0 for no, 1 for yes (remove them).")
  parser.add_argument("--lorelei_embds", type=int, default=0, help="Whether to use multilingual embeddings released for LORELEI. This requires cleaning up words since wordsare are prefixed with the language. 0 for no, 1 for yes (Default is 0)")

  # training
  parser.add_argument("--n_epochs", type=int, default=20)
  parser.add_argument("--batch_size", type=int, default=64)
  parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
  parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
  parser.add_argument("--nonlinear_fc", type=float, default=0, help="use nonlinearity in fc")
  parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
  parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
  parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
  parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
  parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")
  parser.add_argument("--adv_lambda", type=float, default=1., help="adversarial loss weight for hyp only classifier")
  parser.add_argument("--adv_hyp_encoder_lambda", type=float, default=1., help="adversarial weight for hypothesis only encoder")
  parser.add_argument("--nli_net_adv_hyp_encoder_lambda", type=float, default=1., help="adversarial weight for hypothesis encoder in NLI net")
  parser.add_argument("--random_premise_frac", type=float, default=0., help="fraction of random premises to use in NLI net")

  # model
  parser.add_argument("--encoder_type", type=str, default='BLSTMEncoder', help="see list of encoders")
  parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
  parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
  parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
  parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
  parser.add_argument("--pool_type", type=str, default='max', help="max or mean")
  parser.add_argument("--pre_trained_model", type=str, default='', help="Path to pre-trained model")
  parser.add_argument("--pre_trained_adv_model", type=str, default='', help="Path to pre-trained hypothesis model")

  # gpu
  parser.add_argument("--gpu_id", type=int, default=-1, help="GPU ID")
  parser.add_argument("--seed", type=int, default=1234, help="seed")


  #misc
  parser.add_argument("--verbose", type=int, default=1, help="Verbose output")

  params, _ = parser.parse_known_args()

  # print parameters passed, and all parameters
  print('\ntogrep : {0}\n'.format(sys.argv[1:]))
  print(params)

  return params

def get_model_configs(params, n_words):
  """
  MODEL
  """
  # model config
  config_nli_model = {
    'n_words'               :  n_words               ,
    'word_emb_dim'          :  params.word_emb_dim   ,
    'enc_lstm_dim'          :  params.enc_lstm_dim   ,
    'n_enc_layers'          :  params.n_enc_layers   ,
    'dpout_model'           :  params.dpout_model    ,
    'dpout_fc'              :  params.dpout_fc       ,
    'fc_dim'                :  params.fc_dim         ,
    'bsize'                 :  params.batch_size     ,
    'n_classes'             :  params.n_classes      ,
    'pool_type'             :  params.pool_type      ,
    'nonlinear_fc'          :  params.nonlinear_fc   ,
    'encoder_type'          :  params.encoder_type   ,
    'use_cuda'              :  params.gpu_id > -1    ,
    'verbose'               :  params.verbose > 0    ,
    'adv_hyp_encoder_lambda': params.adv_hyp_encoder_lambda,
    'nli_net_adv_hyp_encoder_lambda': params.nli_net_adv_hyp_encoder_lambda,
    'random_premise_frac'   : params.random_premise_frac  ,
  }

  # model
  encoder_types = ['BLSTMEncoder']
                 #, 'BLSTMprojEncoder', 'BGRUlastEncoder',
                 #'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                 #'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']
  assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                             str(encoder_types)
  return config_nli_model

def trainepoch(epoch, train, optimizer_nli, optimizer_hypoth, params, word_vec, shared_nli_net, shared_hypoth_net, \
               loss_fn_nli, loss_fn_hypoth, adv_lambda, adv_hyp_encoder_lambda):
  print('\nTRAINING : Epoch ' + str(epoch))
  shared_nli_net.train()
  shared_hypoth_net.train()
  all_costs_nli, all_costs_hypoth = [], []
  logs = []
  words_count = 0

  last_time = time.time()
  correct_nli, correct_hypoth = 0., 0.
  # shuffle the data
  permutation = np.random.permutation(len(train['hypoths']))

  hypoths, premises, target = [], [], [] 
  for i in permutation:
    hypoths.append(train['hypoths'][i])
    premises.append(train['premises'][i])
    target.append(train['lbls'][i])

  optimizer_nli.param_groups[0]['lr'] = optimizer_nli.param_groups[0]['lr'] * params.decay if epoch>1\
      and 'sgd' in params.optimizer else optimizer_nli.param_groups[0]['lr']
  print('Learning rate (nli) : {0}'.format(optimizer_nli.param_groups[0]['lr']))
  optimizer_hypoth.param_groups[0]['lr'] = optimizer_hypoth.param_groups[0]['lr'] * params.decay if epoch>1\
      and 'sgd' in params.optimizer else optimizer_hypoth.param_groups[0]['lr']
  print('Learning rate (hypoth) : {0}'.format(optimizer_hypoth.param_groups[0]['lr']))

  #trained_sents = 0

  start_time = time.time()
  for stidx in range(0, len(hypoths), params.batch_size):
    # prepare batch
    hypoths_batch, hypoths_len = get_batch(hypoths[stidx:stidx + params.batch_size], word_vec)
    if np.random.random() > params.random_premise_frac:
      random_premise = False
      premises_batch, premises_len = get_batch(premises[stidx:stidx + params.batch_size], word_vec)
    else:
      random_premise = True
      premises_batch, premises_len = get_batch(np.random.choice(premises, len(hypoths_len), replace=False), word_vec)

    tgt_batch = None
    if params.gpu_id > -1: 
      hypoths_batch = Variable(hypoths_batch.cuda())
      premises_batch = Variable(premises_batch.cuda())
      tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size])).cuda()
    else:
      hypoths_batch = Variable(hypoths_batch)
      premises_batch = Variable(premises_batch)
      tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size]))

    k = hypoths_batch.size(1)  # actual batch size

    # model forward
    output_nli = shared_nli_net((premises_batch, premises_len), (hypoths_batch, hypoths_len), random_premise)
    pred_nli = output_nli.data.max(1)[1]
    correct_nli += pred_nli.long().eq(tgt_batch.data.long()).cpu().sum()
    assert len(pred_nli) == len(hypoths[stidx:stidx + params.batch_size])
    
    if adv_lambda > 0:
      output_hypoth = shared_hypoth_net((hypoths_batch, hypoths_len))
      pred_hypoth = output_hypoth.data.max(1)[1]
      correct_hypoth += pred_hypoth.long().eq(tgt_batch.data.long()).cpu().sum()
      assert len(pred_hypoth) == len(hypoths[stidx:stidx + params.batch_size])

    # loss
    loss_nli = loss_fn_nli(output_nli, tgt_batch)
    all_costs_nli.append(loss_nli.item())
    if adv_lambda > 0:
      loss_hypoth = loss_fn_hypoth(output_hypoth, tgt_batch)
      loss_hypoth *= adv_lambda
      all_costs_hypoth.append(loss_hypoth.item())
    words_count += hypoths_batch.nelement() / params.word_emb_dim

    # backward
    optimizer_nli.zero_grad()
    if adv_lambda > 0:
      optimizer_hypoth.zero_grad()
    loss_nli.backward()
    if adv_lambda > 0:
      loss_hypoth.backward()


    def clip_gradients_and_step(net, k, optimizer):

      # gradient clipping (off by default)
      shrink_factor = 1
      total_norm = 0

      for p in net.parameters():
        if p.requires_grad:
          p.grad.data.div_(k)  # divide by the actual batch size
          total_norm += p.grad.data.norm() ** 2
      total_norm = np.sqrt(total_norm)

      if total_norm > params.max_norm:
          shrink_factor = params.max_norm / total_norm
      current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)

      optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update
      
      # optimizer step
      optimizer.step()
      optimizer.param_groups[0]['lr'] = current_lr

    
    clip_gradients_and_step(shared_nli_net, k, optimizer_nli)
    if adv_lambda > 0:
      clip_gradients_and_step(shared_hypoth_net, k, optimizer_hypoth)


    if len(all_costs_nli) == 100:
      logs.append('{0} ; loss nli {1} ; loss hypoth {2} ; sentence/s {3} ; words/s {4} ; accuracy train nli {5} ; accuracy train hypoth {6}'.format(
                            stidx, round(np.mean(all_costs_nli), 2),
                            round(np.mean(all_costs_hypoth), 2),
                            int(len(all_costs_nli) * params.batch_size / (time.time() - last_time)),
                            int(words_count * 1.0 / (time.time() - last_time)),
                            round(100.*correct_nli/(stidx+k), 2), 
                            round(100.*correct_hypoth/(stidx+k),2)))
      print(logs[-1])
      last_time = time.time()
      words_count = 0
      all_costs_nli, all_costs_hypoth = [], []

  train_acc_nli = round(100. * correct_nli/len(hypoths), 2)
  train_acc_hypoth = round(100. * correct_hypoth/len(hypoths), 2)
  print('results : epoch {0} ; mean accuracy train nli : {1}, loss nli : {2} ; mean accuracy train hypoth : {3}, loss hypoth : {4}'
          .format(epoch, train_acc_nli, round(np.mean(all_costs_nli), 2), train_acc_hypoth, round(np.mean(all_costs_hypoth), 2)))
  return train_acc_nli, train_acc_hypoth, shared_nli_net, shared_hypoth_net

def evaluate(epoch, valid, optimizer_nli, optimizer_hypoth, params, word_vec, shared_nli_net, shared_hypoth_net, eval_type='valid', final_eval=False, adv_lambda=1.):
  shared_nli_net.eval()
  shared_hypoth_net.eval()
  correct_nli, correct_hypoth = 0., 0.
  global val_acc_best, lr, stop_training, adam_stop

  if eval_type == 'valid':
    print('\n{0} : Epoch {1}'.format(eval_type, epoch))

  hypoths = valid['hypoths'] #if eval_type == 'valid' else test['s1']
  premises = valid['premises'] #if eval_type == 'valid' else test['s2']
  target = valid['lbls']

  for i in range(0, len(hypoths), params.batch_size):
    # prepare batch
    hypoths_batch, hypoths_len = get_batch(hypoths[i:i + params.batch_size], word_vec)
    premises_batch, premises_len = get_batch(premises[i:i + params.batch_size], word_vec)
    tgt_batch = None
    if params.gpu_id > -1:
      hypoths_batch = Variable(hypoths_batch.cuda())
      premises_batch = Variable(premises_batch.cuda())
      tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).cuda()
    else:
      hypoths_batch = Variable(hypoths_batch)
      premises_batch = Variable(premises_batch)
      tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size]))

    # model forward
    output_nli = shared_nli_net((premises_batch, premises_len), (hypoths_batch, hypoths_len))
    pred_nli = output_nli.data.max(1)[1]
    correct_nli += pred_nli.long().eq(tgt_batch.data.long()).cpu().sum()
    if adv_lambda > 0:
      output_hypoth = shared_hypoth_net((hypoths_batch, hypoths_len))
      pred_hypoth = output_hypoth.data.max(1)[1]
      correct_hypoth += pred_hypoth.long().eq(tgt_batch.data.long()).cpu().sum()

  # save model
  eval_acc_nli = round(100. * correct_nli / len(hypoths), 2)
  eval_acc_hypoth = round(100. * correct_hypoth / len(hypoths), 2)
  if final_eval:
    print('finalgrep : accuracy {0} : nli {1}, hypoth {2}'.format(eval_type, eval_acc_nli, eval_acc_hypoth))
  else:
    print('togrep : results : epoch {0} ; mean accuracy {1} : nli {2}, hypoth {3}'.format(epoch, eval_type, eval_acc_nli, eval_acc_hypoth))

  if eval_type == 'valid' and epoch <= params.n_epochs:
    if eval_acc_nli > val_acc_best:
      print('saving model at epoch {0}'.format(epoch))
      if not os.path.exists(params.outputdir):
        os.makedirs(params.outputdir)
      torch.save(shared_nli_net, os.path.join(params.outputdir, params.outputmodelname))
      torch.save(shared_hypoth_net, os.path.join(params.outputdir, params.outputmodelname_adversary))
      val_acc_best = eval_acc_nli
    else:
      if 'sgd' in params.optimizer:
        optimizer_nli.param_groups[0]['lr'] = optimizer_nli.param_groups[0]['lr'] / params.lrshrink
        print('Shrinking nli lr by : {0}. New lr = {1}'.format(params.lrshrink,
                              optimizer_nli.param_groups[0]['lr']))
        optimizer_hypoth.param_groups[0]['lr'] = optimizer_hypoth.param_groups[0]['lr'] / params.lrshrink
        print('Shrinking hypoth lr by : {0}. New lr = {1}'.format(params.lrshrink,
                              optimizer_hypoth.param_groups[0]['lr']))
        if optimizer_nli.param_groups[0]['lr'] < params.minlr:
          stop_training = True
      if 'adam' in params.optimizer:
        # early stopping (at 2nd decrease in accuracy)
        stop_training = adam_stop
        adam_stop = True
  return eval_acc_nli, eval_acc_hypoth

def main(args):
  print "main"

  """
  SEED
  """
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.gpu_id > -1:
    torch.cuda.manual_seed(args.seed)

  """
  DATA
  """
  train, val, test = get_nli_text(args.train_lbls_file, args.train_src_file, args.val_lbls_file, \
                                    args.val_src_file, args.test_lbls_file, args.test_src_file, \
                                    args.max_train_sents, args.max_val_sents, args.max_test_sents, args.remove_dup)

  word_vecs = build_vocab(train['hypoths'] + val['hypoths'] + test['hypoths'] + train['premises'] + val['premises'] + test['premises'] , args.embdfile, args.lorelei_embds)
  args.word_emb_dim = len(word_vecs[word_vecs.keys()[0]])

  nli_model_configs = get_model_configs(args, len(word_vecs))

  lbls_file = args.train_lbls_file
  if "mpe" in lbls_file or "snli" in lbls_file or "multinli" in lbls_file or "sick" in lbls_file or "joci" in lbls_file or "glue" in lbls_file:
    nli_model_configs["n_classes"] = 3
  elif "spr" in lbls_file or "dpr" in lbls_file or "fnplus" in lbls_file or "add_one" in lbls_file or "scitail" in lbls_file:
    nli_model_configs["n_classes"] = 2

  # define premise and hypoth encoders
  premise_encoder = eval(nli_model_configs['encoder_type'])(nli_model_configs)
  hypoth_encoder = eval(nli_model_configs['encoder_type'])(nli_model_configs)
  shared_nli_net = SharedNLINet(nli_model_configs, premise_encoder, hypoth_encoder)
  shared_hypoth_net = SharedHypothNet(nli_model_configs, hypoth_encoder)
  print(shared_nli_net)
  print(shared_hypoth_net)

  if args.pre_trained_model:
    print "Pre_trained_model: " + args.pre_trained_model
    pre_trained_model = torch.load(args.pre_trained_model)
  
    shared_nli_net_params = shared_nli_net.state_dict()
    pre_trained_params = pre_trained_model.state_dict()
    assert shared_nli_net_params.keys() == pre_trained_params.keys(), "load model has different parameter state names that NLI_HYPOTHS_NET"
    for key, parameters in shared_nli_net_params.items():
      if parameters.size() == pre_trained_params[key].size():
        shared_nli_net_params[key] = pre_trained_params[key]
    shared_nli_net.load_state_dict(shared_nli_net_params)

  print(shared_nli_net)

  if args.pre_trained_adv_model:
    print "Pre_trained_adv_model: " + args.pre_trained_adv_model
    pre_trained_model = torch.load(args.pre_trained_adv_model)
  
    shared_hypoth_net_params = shared_hypoth_net.state_dict()
    pre_trained_params = pre_trained_model.state_dict()
    assert shared_hypoth_net_params.keys() == pre_trained_params.keys(), "load model has different parameter state names that NLI_HYPOTHS_NET"
    for key, parameters in nli_hypoth_params.items():
      if parameters.size() == pre_trained_params[key].size():
        shared_hypoth_net_params[key] = pre_trained_params[key]
    shared_hypoth_net.load_state_dict(shared_hypoth_net_params)

  print(shared_hypoth_net)


  # nli loss
  weight = torch.FloatTensor(args.n_classes).fill_(1)
  loss_fn_nli = nn.CrossEntropyLoss(weight=weight)
  loss_fn_nli.size_average = False

  # hypoth (adversarial) loss
  weight = torch.FloatTensor(args.n_classes).fill_(1)
  loss_fn_hypoth = nn.CrossEntropyLoss(weight=weight)
  loss_fn_hypoth.size_average = False

  # optimizer
  optim_fn, optim_params = get_optimizer(args.optimizer)
  optimizer_nli = optim_fn(shared_nli_net.parameters(), **optim_params)
  #optimizer_hypoth = optim_fn(shared_hypoth_net.parameters(), **optim_params)
  # only pass hypoth classifier params to avoid updating shared encoder params twice 
  optimizer_hypoth = optim_fn(shared_hypoth_net.classifier.parameters(), **optim_params)

  if args.gpu_id > -1:
    shared_nli_net.cuda()
    shared_hypoth_net.cuda()
    loss_fn_nli.cuda()
    loss_fn_hypoth.cuda()

  """
  TRAIN
  """
  global val_acc_best, lr, stop_training, adam_stop
  val_acc_best = -1e10
  adam_stop = False
  stop_training = False
  lr = optim_params['lr'] if 'sgd' in args.optimizer else None

  """
  Train model on Natural Language Inference task
  """
  epoch = 1

  while not stop_training and epoch <= args.n_epochs:
    train_acc_nli, train_acc_hypoth, shared_nli_net, shared_hypoth_net = trainepoch(epoch, train, optimizer_nli, optimizer_hypoth, args, word_vecs, shared_nli_net, shared_hypoth_net, loss_fn_nli, loss_fn_hypoth, args.adv_lambda, args.adv_hyp_encoder_lambda)
    eval_acc_nli, eval_acc_hypoth = evaluate(epoch, val, optimizer_nli, optimizer_hypoth, args, word_vecs, shared_nli_net, shared_hypoth_net, 'valid', adv_lambda=args.adv_lambda)
    epoch += 1


if __name__ == '__main__':
  args = get_args()
  main(args)
