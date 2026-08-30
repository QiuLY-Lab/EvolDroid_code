import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc = [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)

      pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors)

      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))

  return np.mean(val_ap), np.mean(val_auc)


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors,node_features,device):
  # pred_prob = np.zeros(len(data.sources))
  num_instance = len(data.sources)
  num_classes = 25  # 或根据 decoder 输出自动推断
  pred_prob = np.zeros((num_instance, num_classes))

  num_batch = math.ceil(num_instance / batch_size)
  pred_classes = np.zeros(num_instance, dtype=int)

  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = edge_idxs[s_idx: e_idx]

      # source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
      #                                                                              destinations_batch,
      #                                                                              destinations_batch,
      #                                                                              timestamps_batch,
      #                                                                              edge_idxs_batch,
      #                                                                              n_neighbors)
      # 不使用tgn 而是直接取节点原始特征
      # print("不使用tgn")
      source_embedding = torch.from_numpy(node_features[destinations_batch]).float().to(device)
      # source_embedding = destination_embedding

      # pred_prob_batch = decoder(source_embedding).sigmoid()
      # pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

      # 🧠 分类器预测 logits
      logits_batch = decoder(source_embedding)  # [batch_size, num_classes]
      probs_batch = torch.softmax(logits_batch, dim=1).cpu().numpy()
      preds_batch = np.argmax(probs_batch, axis=1)

      # 存储预测结果
      pred_prob[s_idx:e_idx, :] = probs_batch
      pred_classes[s_idx:e_idx] = preds_batch

  from sklearn.preprocessing import label_binarize
  y_true_bin = label_binarize(data.labels, classes=range(25))
  auc_roc = roc_auc_score(y_true_bin, pred_prob)

  from sklearn.metrics import accuracy_score, f1_score
  acc = accuracy_score(data.labels, pred_classes)
  f1 = f1_score(data.labels, pred_classes, average='macro')
  f11 = f1_score(data.labels, pred_classes, average=None)
  print("acc",acc)
  print("f1",f1)
  print("f11",f11)
  print("f11.shape",f11.shape)

  # auc_roc = roc_auc_score(data.labels, pred_prob)
  return auc_roc,acc,f1,f11
