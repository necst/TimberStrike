//
// Created by liqinbin on 10/14/20.
//

#include "FedTree/Tree/gbdt.h"
#include "FedTree/booster.h"
#include "FedTree/FL/partition.h"


void GBDT::train(GBDTParam &param, DataSet &dataset) {
    if (param.tree_method == "auto")
        param.tree_method = "hist";
    else if (param.tree_method != "hist") {
        std::cout << "FedTree only supports histogram-based training yet";
        exit(1);
    }

    if (param.objective.find("multi:") != std::string::npos || param.objective.find("binary:") != std::string::npos) {
        int num_class = dataset.label.size();
        if (param.num_class != num_class) {
            LOG(INFO) << "updating number of classes from " << param.num_class << " to " << num_class;
            param.num_class = num_class;
        }
        if (param.num_class > 2)
            param.tree_per_round = param.num_class;
    } else if (param.objective.find("reg:") != std::string::npos) {
        param.num_class = 1;
    }

    Booster booster;
    booster.init(dataset, param);
    std::chrono::high_resolution_clock timer;
    auto start = timer.now();
    for (int i = 0; i < param.n_trees; ++i) {
        //one iteration may produce multiple trees, depending on objectives
        booster.boost(trees);
    }

//    float_type score = predict_score(param, dataset);
//    LOG(INFO) << score;

    auto stop = timer.now();
    std::chrono::duration<float> training_time = stop - start;
    LOG(INFO) << "training time = " << training_time.count();
    return;
}

void GBDT::train_a_subtree(GBDTParam &param, DataSet &dataset, int n_layer, int *id_list, int *nins_list, float_type *gradient_g_list, 
                            float_type *gradient_h_list, int *n_node, int *node_id_list, float_type *input_gradient_g, float_type *input_gradient_h) {
    if (param.tree_method == "auto")
        param.tree_method = "hist";
    else if (param.tree_method != "hist") {
        std::cout << "FedTree only supports histogram-based training yet";
        exit(1);
    }

    if (param.objective.find("multi:") != std::string::npos || param.objective.find("binary:") != std::string::npos) {
        int num_class = dataset.label.size();
        if (param.num_class != num_class) {
            LOG(INFO) << "updating number of classes from " << param.num_class << " to " << num_class;
            param.num_class = num_class;
        }
        if (param.num_class > 2)
            param.tree_per_round = param.num_class;
    } else if (param.objective.find("reg:") != std::string::npos) {
        param.num_class = 1;
    }

    Booster booster;
    booster.init(dataset, param);
    std::chrono::high_resolution_clock timer;
    auto start = timer.now();
    std::cout<<"start boost a subtree"<<std::endl;
    booster.boost_a_subtree(trees, n_layer, id_list, nins_list, gradient_g_list, gradient_h_list, n_node, node_id_list, input_gradient_g, input_gradient_h);
    //booster.boost(trees);
//    float_type score = predict_score(param, dataset);
//    LOG(INFO) << score;

    auto stop = timer.now();
    std::chrono::duration<float> training_time = stop - start;
    LOG(INFO) << "training time = " << training_time.count();
    return;
}


vector<float_type> GBDT::predict(const GBDTParam &model_param, const DataSet &dataSet) {
    SyncArray<float_type> y_predict;
    predict_raw(model_param, dataSet, y_predict);
    //convert the aggregated values to labels, probabilities or ranking scores.
    std::unique_ptr<ObjectiveFunction> obj;
    obj.reset(ObjectiveFunction::create(model_param.objective));
    obj->configure(model_param, dataSet);
    //compute metric
    std::unique_ptr<Metric> metric;
    metric.reset(Metric::create(obj->default_metric_name()));
    metric->configure(model_param, dataSet);
    LOG(INFO) << metric->get_name().c_str() << " = " << metric->get_score(y_predict);

    obj->predict_transform(y_predict);
    vector<float_type> y_pred_vec(y_predict.size());
    memcpy(y_pred_vec.data(), y_predict.host_data(), sizeof(float_type) * y_predict.size());
    return y_pred_vec;
}

// for vertical FL prediction
vector<float_type> GBDT::predict(const GBDTParam &model_param, const vector<DataSet> &dataSet) {
    SyncArray<float_type> y_predict;
    predict_raw_vertical(model_param, dataSet, y_predict);
    //convert the aggregated values to labels, probabilities or ranking scores.
    std::unique_ptr<ObjectiveFunction> obj;
    obj.reset(ObjectiveFunction::create(model_param.objective));
    obj->configure(model_param, dataSet[0]);
    //compute metric
    std::unique_ptr<Metric> metric;
    metric.reset(Metric::create(obj->default_metric_name()));
    metric->configure(model_param, dataSet[0]);
    LOG(INFO) << metric->get_name().c_str() << " = " << metric->get_score(y_predict);

    obj->predict_transform(y_predict);
    vector<float_type> y_pred_vec(y_predict.size());
    memcpy(y_pred_vec.data(), y_predict.host_data(), sizeof(float_type) * y_predict.size());
    return y_pred_vec;
}


// private helper function for auc calculation
float_type calculate_auc(const vector<float_type> &yp_data, const vector<float_type> &y_data) {
    int n = yp_data.size();
    std::vector<std::pair<double, int>> prediction_labels(n);
    for (int i = 0; i < n; ++i) {
        prediction_labels[i] = std::make_pair(yp_data[i], y_data[i]);
    }
    std::sort(prediction_labels.begin(), prediction_labels.end(),
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                  return a.first > b.first;
              });

    double auc = 0.0;
    long pos_count = 0;
    long neg_count = 0;
    for (int i = 0; i < n; ++i) {
        if (prediction_labels[i].second == 1) {
            ++pos_count;
        } else {
            auc += pos_count;
            ++neg_count;
        }
    }
    if (pos_count == 0 || neg_count == 0) {
        std::cerr << "Warning: Only one class present in the data." << std::endl;
        return -1;
    }
    auc /= (pos_count * neg_count);
    return auc;
}

float_type GBDT::predict_score(const GBDTParam &model_param, const DataSet &dataSet) {
    SyncArray<float_type> y_predict;
    predict_raw(model_param, dataSet, y_predict);
    LOG(DEBUG) << "y_predict:" << y_predict;
    //convert the aggregated values to labels, probabilities or ranking scores.
    std::unique_ptr<ObjectiveFunction> obj;
    obj.reset(ObjectiveFunction::create(model_param.objective));
    obj->configure(model_param, dataSet);

    //compute metric
    std::unique_ptr<Metric> metric;
    metric.reset(Metric::create(obj->default_metric_name()));
    metric->configure(model_param, dataSet);
    float_type score = metric->get_score(y_predict);

//    LOG(INFO) << metric->get_name().c_str() << " = " << score;
    LOG(INFO) << metric->get_name() << " = " << score;

    if (metric->get_name() == "F1") {
        // calculate AUC
        auto label = dataSet.y;
        float_type* yp_data = y_predict.host_data();
        vector<float_type> yp_vec(yp_data, yp_data + y_predict.size());
        LOG(DEBUG) << "yp_vec:" << yp_vec;
        LOG(DEBUG) << "label:" << label;
        float_type auc = calculate_auc(yp_vec, label);
        LOG(INFO) << "AUC = " << auc;
    }
    return score;
}

float_type GBDT::predict_score_vertical(const GBDTParam &model_param, const DataSet &dataSet,
                                        std::map<int, vector<int>> &batch_idxs) {
    SyncArray<float_type> y_predict;
    predict_raw_vertical(model_param, dataSet, y_predict, batch_idxs);
    //LOG(INFO) << "y_predict:" << y_predict;
    //convert the aggregated values to labels, probabilities or ranking scores.
    std::unique_ptr<ObjectiveFunction> obj;
    obj.reset(ObjectiveFunction::create(model_param.objective));
    obj->configure(model_param, dataSet);

    //compute metric
    std::unique_ptr<Metric> metric;
    metric.reset(Metric::create(obj->default_metric_name()));
    metric->configure(model_param, dataSet);
    float_type score = metric->get_score(y_predict);

//    LOG(INFO) << metric->get_name().c_str() << " = " << score;
    LOG(INFO) << metric->get_name() << " = " << score;
    return score;
}

float_type GBDT::predict_score_vertical(const GBDTParam &model_param, const vector<DataSet> &dataSet) {
    SyncArray<float_type> y_predict;
    predict_raw_vertical(model_param, dataSet, y_predict);
//    LOG(INFO) << "y_predict:" << y_predict;
    //convert the aggregated values to labels, probabilities or ranking scores.
    std::unique_ptr<ObjectiveFunction> obj;
    obj.reset(ObjectiveFunction::create(model_param.objective));
    obj->configure(model_param, dataSet[0]);

    //compute metric
    std::unique_ptr<Metric> metric;
    metric.reset(Metric::create(obj->default_metric_name()));
    metric->configure(model_param, dataSet[0]);
    float_type score = metric->get_score(y_predict);

//    LOG(INFO) << metric->get_name().c_str() << " = " << score;
    LOG(INFO) << metric->get_name() << " = " << score;
    return score;
}

void GBDT::predict_raw(const GBDTParam &model_param, const DataSet &dataSet, SyncArray<float_type> &y_predict) {
    TIMED_SCOPE(timerObj, "predict");
    int n_instances = dataSet.n_instances();
//    int n_features = dataSet.n_features();

    //the whole model to an array
    int num_iter = trees.size();
    int num_class = trees.front().size();
    int num_node = trees[0][0].nodes.size();

    int total_num_node = num_iter * num_class * num_node;
    y_predict.resize(n_instances * num_class);
    SyncArray<Tree::TreeNode> model(total_num_node);
    auto model_data = model.host_data();
    int tree_cnt = 0;
    for (auto &vtree:trees) {
        for (auto &t:vtree) {
            memcpy(model_data + num_node * tree_cnt, t.nodes.host_data(), sizeof(Tree::TreeNode) * num_node);
            tree_cnt++;
        }
    }

    PERFORMANCE_CHECKPOINT_WITH_ID(timerObj, "init trees");

    //do prediction
    auto model_host_data = model.host_data();
    auto predict_data = y_predict.host_data();
    auto csr_col_idx_data = dataSet.csr_col_idx.data();
    auto csr_val_data = dataSet.csr_val.data();
    auto csr_row_ptr_data = dataSet.csr_row_ptr.data();
    auto lr = model_param.learning_rate;
    PERFORMANCE_CHECKPOINT_WITH_ID(timerObj, "copy data");

    //predict BLOCK_SIZE instances in a block, 1 thread for 1 instance
//    int BLOCK_SIZE = 128;
    //determine whether we can use shared memory
//    size_t smem_size = n_features * BLOCK_SIZE * sizeof(float_type);
//    int NUM_BLOCK = (n_instances - 1) / BLOCK_SIZE + 1;

    //use sparse format and binary search
#pragma omp parallel for
    for (int iid = 0; iid < n_instances; iid++) {
        auto get_next_child = [&](Tree::TreeNode node, float_type feaValue) {
            //return feaValue < node.split_value ? node.lch_index : node.rch_index;
            return (feaValue - node.split_value) >= -1e-6 ? node.rch_index : node.lch_index;
        };
        auto get_val = [&](const int *row_idx, const float_type *row_val, int row_len, int idx,
                           bool *is_missing) -> float_type {
            //binary search to get feature value
            const int *left = row_idx;
            const int *right = row_idx + row_len;

            while (left != right) {
                const int *mid = left + (right - left) / 2;
                if (*mid == idx) {
                    *is_missing = false;
                    return row_val[mid - row_idx];
                }
                if (*mid > idx)
                    right = mid;
                else left = mid + 1;
            }
            *is_missing = true;
            return 0;
        };
        const int *col_idx = csr_col_idx_data + csr_row_ptr_data[iid];
        const float_type *row_val = csr_val_data + csr_row_ptr_data[iid];
        int row_len = csr_row_ptr_data[iid + 1] - csr_row_ptr_data[iid];
        for (int t = 0; t < num_class; t++) {
            auto predict_data_class = predict_data + t * n_instances;
            float_type sum = 0;
            for (int iter = 0; iter < num_iter; iter++) {
                const Tree::TreeNode *node_data = model_host_data + iter * num_class * num_node + t * num_node;
                Tree::TreeNode curNode = node_data[0];
                int cur_nid = 0; //node id
                while (!curNode.is_leaf) {
                    int fid = curNode.split_feature_id;
                    bool is_missing;
                    float_type fval = get_val(col_idx, row_val, row_len, fid, &is_missing);
                    if (!is_missing)
                        cur_nid = get_next_child(curNode, fval);
                    else if (curNode.default_right)
                        cur_nid = curNode.rch_index;
                    else
                        cur_nid = curNode.lch_index;
                    curNode = node_data[cur_nid];
                }
                sum += lr * curNode.base_weight;
            }
            if (model_param.bagging)
                sum /= num_iter;
            predict_data_class[iid] += sum;
        }//end all tree prediction
    }
}

void GBDT::predict_raw_vertical(const GBDTParam &model_param, const DataSet &dataSet, SyncArray<float_type> &y_predict,
                                std::map<int, vector<int>> &batch_idxs) {
    TIMED_SCOPE(timerObj, "predict");

    vector<int> idx_map;
    if(!batch_idxs.empty()) {
        for (int i = 0; i < batch_idxs.size(); i++) {
            for (int idx: batch_idxs[i]) {
                idx_map.push_back(idx);
            }
        }
    }

    int n_instances = dataSet.n_instances();
//    int n_features = dataSet.n_features();

    //the whole model to an array
    int num_iter = trees.size();
    int num_class = trees.front().size();
    int num_node = trees[0][0].nodes.size();
    int total_num_node = num_iter * num_class * num_node;
    //TODO: reduce the output size for binary classification
    y_predict.resize(n_instances * num_class);

    SyncArray<Tree::TreeNode> model(total_num_node);
    auto model_data = model.host_data();
    int tree_cnt = 0;
    for (auto &vtree:trees) {
        for (auto &t:vtree) {
            memcpy(model_data + num_node * tree_cnt, t.nodes.host_data(), sizeof(Tree::TreeNode) * num_node);
            tree_cnt++;
        }
    }

    PERFORMANCE_CHECKPOINT_WITH_ID(timerObj, "init trees");

    //do prediction
    auto model_host_data = model.host_data();
    auto predict_data = y_predict.host_data();
    auto csr_col_idx_data = dataSet.csr_col_idx.data();
    auto csr_val_data = dataSet.csr_val.data();
    auto csr_row_ptr_data = dataSet.csr_row_ptr.data();
    auto lr = model_param.learning_rate;
    PERFORMANCE_CHECKPOINT_WITH_ID(timerObj, "copy data");

    //predict BLOCK_SIZE instances in a block, 1 thread for 1 instance
//    int BLOCK_SIZE = 128;
    //determine whether we can use shared memory
//    size_t smem_size = n_features * BLOCK_SIZE * sizeof(float_type);
//    int NUM_BLOCK = (n_instances - 1) / BLOCK_SIZE + 1;

    //use sparse format and binary search
#pragma omp parallel for
    for (int iid = 0; iid < n_instances; iid++) {
        auto get_next_child = [&](Tree::TreeNode node, float_type feaValue) {
            //return feaValue < node.split_value ? node.lch_index : node.rch_index;
            return (feaValue - node.split_value) >= -1e-6 ? node.rch_index : node.lch_index;
        };
        auto get_val = [&](const int *row_idx, const float_type *row_val, int row_len, int idx,
                           bool *is_missing) -> float_type {
            //binary search to get feature value
            const int *left = row_idx;
            const int *right = row_idx + row_len;

            while (left != right) {
                const int *mid = left + (right - left) / 2;
                if (*mid == idx) {
                    *is_missing = false;
                    return row_val[mid - row_idx];
                }
                if (*mid > idx)
                    right = mid;
                else left = mid + 1;
            }
            *is_missing = true;
            return 0;
        };
        const int *col_idx = csr_col_idx_data + csr_row_ptr_data[iid];
        const float_type *row_val = csr_val_data + csr_row_ptr_data[iid];
        int row_len = csr_row_ptr_data[iid + 1] - csr_row_ptr_data[iid];
        for (int t = 0; t < num_class; t++) {
            auto predict_data_class = predict_data + t * n_instances;
            float_type sum = 0;
            for (int iter = 0; iter < num_iter; iter++) {
                const Tree::TreeNode *node_data = model_host_data + iter * num_class * num_node + t * num_node;
                Tree::TreeNode curNode = node_data[0];
                int cur_nid = 0; //node id
                while (!curNode.is_leaf) {
                    int fid;
                    if(!batch_idxs.empty())
                        fid = idx_map[curNode.split_feature_id];
                    else
                        fid = curNode.split_feature_id;
                    bool is_missing;
                    float_type fval = get_val(col_idx, row_val, row_len, fid, &is_missing);
                    if (!is_missing)
                        cur_nid = get_next_child(curNode, fval);
                    else if (curNode.default_right)
                        cur_nid = curNode.rch_index;
                    else
                        cur_nid = curNode.lch_index;
                    curNode = node_data[cur_nid];
                }
                sum += lr * curNode.base_weight;
                if (model_param.bagging)
                    sum /= num_iter;
            }
            predict_data_class[iid] += sum;
        }//end all tree prediction
    }
}

void GBDT::predict_raw_vertical(const GBDTParam &model_param, const vector<DataSet> &dataSet, SyncArray<float_type> &y_predict) {
    TIMED_SCOPE(timerObj, "predict");
    int n_parties = dataSet.size();
    int n_instances = dataSet[0].n_instances();

    //the whole model to an array
    int num_iter = trees.size();
    int num_class = trees.front().size();
    int num_node = trees[0][0].nodes.size();
    int total_num_node = num_iter * num_class * num_node;
    //TODO: reduce the output size for binary classification
    y_predict.resize(n_instances * num_class);

    SyncArray<Tree::TreeNode> model(total_num_node);
    auto model_data = model.host_data();
    int tree_cnt = 0;
    for (auto &vtree:trees) {
        for (auto &t:vtree) {
            memcpy(model_data + num_node * tree_cnt, t.nodes.host_data(), sizeof(Tree::TreeNode) * num_node);
            tree_cnt++;
        }
    }

    PERFORMANCE_CHECKPOINT_WITH_ID(timerObj, "init trees");

    //do prediction
    auto model_host_data = model.host_data();
    auto predict_data = y_predict.host_data();
    vector<const int *> csr_col_idx(n_parties);
    vector<const float_type *> csr_val_data(n_parties);
    vector<const int *> csr_row_ptr(n_parties);
    for(int i = 0; i < n_parties; i++){
        csr_col_idx[i] = dataSet[i].csr_col_idx.data();
        csr_val_data[i] = dataSet[i].csr_val.data();
        csr_row_ptr[i] = dataSet[i].csr_row_ptr.data();
    }

    auto lr = model_param.learning_rate;
    PERFORMANCE_CHECKPOINT_WITH_ID(timerObj, "copy data");

    //use sparse format and binary search
#pragma omp parallel for
    for (int iid = 0; iid < n_instances; iid++) {
        auto get_next_child = [&](Tree::TreeNode node, float_type feaValue) {
            //return feaValue < node.split_value ? node.lch_index : node.rch_index;
            return (feaValue - node.split_value) >= -1e-6 ? node.rch_index : node.lch_index;
        };
        auto get_val = [&](const int *row_idx, const float_type *row_val, int row_len, int idx,
                           bool *is_missing) -> float_type {
            //binary search to get feature value
            const int *left = row_idx;
            const int *right = row_idx + row_len;

            while (left != right) {
                const int *mid = left + (right - left) / 2;
                if (*mid == idx) {
                    *is_missing = false;
                    return row_val[mid - row_idx];
                }
                if (*mid > idx)
                    right = mid;
                else left = mid + 1;
            }
            *is_missing = true;
            return 0;
        };
        for (int t = 0; t < num_class; t++) {
            auto predict_data_class = predict_data + t * n_instances;
            float_type sum = 0;
            for (int iter = 0; iter < num_iter; iter++) {
                const Tree::TreeNode *node_data = model_host_data + iter * num_class * num_node + t * num_node;
                Tree::TreeNode curNode = node_data[0];
                int cur_nid = 0; //node id
                while (!curNode.is_leaf) {
                    int fid = curNode.split_feature_id;
                    int pid = 0;
                    for(int i = 0; i < n_parties; i++){
                        if(dataSet[i].n_features() > fid){
                            break;
                        }
                        else{
                            fid -= dataSet[i].n_features();
                            pid++;
                        }
                    }
                    const int* col_idx = csr_col_idx[pid] + csr_row_ptr[pid][iid];
                    const float_type* row_val = csr_val_data[pid] + csr_row_ptr[pid][iid];
                    int row_len = csr_row_ptr[pid][iid+1] - csr_row_ptr[pid][iid];

                    bool is_missing;
                    float_type fval = get_val(col_idx, row_val, row_len, fid, &is_missing);
                    if (!is_missing)
                        cur_nid = get_next_child(curNode, fval);
                    else if (curNode.default_right)
                        cur_nid = curNode.rch_index;
                    else
                        cur_nid = curNode.lch_index;
                    curNode = node_data[cur_nid];
                }
                sum += lr * curNode.base_weight;
                if (model_param.bagging)
                    sum /= num_iter;
            }
            predict_data_class[iid] += sum;
        }//end all tree prediction
    }
}

void GBDT::predict_leaf(const GBDTParam &model_param, const DataSet &dataSet, SyncArray<float_type> &y_predict, int *ins2leaf) {
    TIMED_SCOPE(timerObj, "predict");
    int n_instances = dataSet.n_instances();
//    int n_features = dataSet.n_features();

    //the whole model to an array
    int num_iter = trees.size();
    int num_class = trees.front().size();
    int num_node = trees[0][0].nodes.size();

    int total_num_node = num_iter * num_class * num_node;
    // y_predict.resize(n_instances * num_class);
    std::cout<<"num_class in predict_raw:"<<num_class<<std::endl;
    SyncArray<Tree::TreeNode> model(total_num_node);
    auto model_data = model.host_data();
    int tree_cnt = 0;
    for (auto &vtree:trees) {
        for (auto &t:vtree) {
            memcpy(model_data + num_node * tree_cnt, t.nodes.host_data(), sizeof(Tree::TreeNode) * num_node);
            tree_cnt++;
        }
    }

    PERFORMANCE_CHECKPOINT_WITH_ID(timerObj, "init trees");

    //do prediction
    auto model_host_data = model.host_data();
    // auto predict_data = y_predict.host_data();
    auto csr_col_idx_data = dataSet.csr_col_idx.data();
    auto csr_val_data = dataSet.csr_val.data();
    auto csr_row_ptr_data = dataSet.csr_row_ptr.data();
    auto lr = model_param.learning_rate;
    PERFORMANCE_CHECKPOINT_WITH_ID(timerObj, "copy data");

#pragma omp parallel for
    for (int iid = 0; iid < n_instances; iid++) {
        auto get_next_child = [&](Tree::TreeNode node, float_type feaValue) {
            //return feaValue < node.split_value ? node.lch_index : node.rch_index;
            return (feaValue - node.split_value) >= -1e-6 ? node.rch_index : node.lch_index;
        };
        auto get_val = [&](const int *row_idx, const float_type *row_val, int row_len, int idx,
                           bool *is_missing) -> float_type {
            //binary search to get feature value
            const int *left = row_idx;
            const int *right = row_idx + row_len;

            while (left != right) {
                const int *mid = left + (right - left) / 2;
                if (*mid == idx) {
                    *is_missing = false;
                    return row_val[mid - row_idx];
                }
                if (*mid > idx)
                    right = mid;
                else left = mid + 1;
            }
            *is_missing = true;
            return 0;
        };
        const int *col_idx = csr_col_idx_data + csr_row_ptr_data[iid];
        const float_type *row_val = csr_val_data + csr_row_ptr_data[iid];
        int row_len = csr_row_ptr_data[iid + 1] - csr_row_ptr_data[iid];
        for (int t = 0; t < num_class; t++) {
            // auto predict_data_class = predict_data + t * n_instances;
            // float_type sum = 0;
            for (int iter = 0; iter < num_iter; iter++) {
                const Tree::TreeNode *node_data = model_host_data + iter * num_class * num_node + t * num_node;
                Tree::TreeNode curNode = node_data[0];
                int cur_nid = 0; //node id
                while (!curNode.is_leaf) {
                    int fid = curNode.split_feature_id;
                    bool is_missing;
                    float_type fval = get_val(col_idx, row_val, row_len, fid, &is_missing);
                    if (!is_missing)
                        cur_nid = get_next_child(curNode, fval);
                    else if (curNode.default_right)
                        cur_nid = curNode.rch_index;
                    else
                        cur_nid = curNode.lch_index;

                    curNode = node_data[cur_nid];
                }
                ins2leaf[iter * n_instances + iid] = cur_nid;
                // sum += lr * curNode.base_weight;
            }
            // if (model_param.bagging)
                // sum /= num_iter;
        }//end all tree prediction
    }
}
