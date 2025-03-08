from os import name
import pandas as pd
from itertools import product
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import zscore
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from statistics import mean, stdev
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest


def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

class TransferEncoder:

    set_seeds()
    def __init__(self, n_steps):
        self.n_steps = n_steps
        self.W = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.history = {'train_loss': [], 'val_loss': []}

    #def build_hidden_layer(self, input_dim, n_hidden):
    #    self.hidden_layer_weights = tf.Variable(
    #        tf.random.uniform([input_dim, n_hidden], 
    #                          minval=-1.0 / np.sqrt(input_dim), 
    #                          maxval=1.0 / np.sqrt(input_dim))
    #    )
    #    self.hidden_layer_biases = tf.Variable(tf.zeros([n_hidden]))
    #    self.output_layer_weights = tf.Variable(
    #        tf.random.uniform([n_hidden, input_dim], 
    #                          minval=-1.0 / np.sqrt(n_hidden), 
    #                          maxval=1.0 / np.sqrt(n_hidden))
    #    )
    #    self.built = True

    def build_hidden_layer(self, input_dim, n_hidden):
        self.W = tf.Variable(
            tf.keras.initializers.GlorotUniform(seed = 42)(shape=[input_dim, n_hidden]),
            name='W'
        )
        self.b1 = tf.Variable(tf.zeros([n_hidden]), name='b1')
        
        self.W2 = tf.Variable(
            tf.keras.initializers.GlorotUniform(seed=42)(shape=[n_hidden, input_dim]),
            name='W2'
        )
        self.b2 = tf.Variable(tf.zeros([input_dim]), name='b2')
        self.built = True

    #@tf.function
    def call(self, inputs):
        hidden_layer = tf.nn.sigmoid(tf.matmul(inputs, self.W) + self.b1)
        #W2 = tf.transpose(self.W)
        z_hat = tf.nn.sigmoid(tf.matmul(hidden_layer, self.W2) + self.b2)
        return z_hat

    #@tf.function
    def train_step(self, x_batch, z_batch, optimizer):
        with tf.GradientTape() as tape:
            z_hat = self.call(x_batch)
            cost = tf.reduce_mean(tf.keras.losses.binary_crossentropy(z_batch, z_hat))
            l2_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in [self.W, self.b1,self.W2, self.b2]])
            cost += l2_loss
        gradients = tape.gradient(cost, [self.W, self.b1,self.W2, self.b2])
        optimizer.apply_gradients(zip(gradients, [self.W, self.b1,self.W2, self.b2]))
        return cost

    #@tf.function
    def validation_step(self, x_batch, z_batch):
        z_hat = self.call(x_batch)
        cost = tf.reduce_mean(tf.keras.losses.binary_crossentropy(z_batch, z_hat))
        l2_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in [self.W, self.b1,self.W2, self.b2]])
        cost += l2_loss
        return cost

    def fit_layers(self, X_train, Z_train, X_val, Z_val, batch_size=float('inf')):
        hidden_units_range = range(50, self.n_steps + 1, 50)
        best_config = {
            "n_hidden": None,
            "best_epoch": 0,
            "best_loss_val": float('inf'),
            "best_weights": None,
            "train_loss_history": None,
            "val_loss_history": None
        }

        X_train = X_train.astype(np.float32)
        Z_train = Z_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        Z_val = Z_val.astype(np.float32)

        for n_hidden in hidden_units_range:
            input_dim = X_train.shape[1]
            self.build_hidden_layer(input_dim, n_hidden)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

            n_samples = len(X_train)
           
            idx = tf.random.shuffle(tf.range(n_samples))
            X = tf.gather(X_train, idx)
            Z = tf.gather(Z_train, idx)

            n_samples_val = len(X_val)
            idx_val = tf.random.shuffle(tf.range(n_samples_val))
            X_val = tf.gather(X_val, idx_val)
            Z_val = tf.gather(Z_val, idx_val)

            batch_size = min(n_samples, batch_size)
            batch_size_val = min(n_samples_val, batch_size)

            best_loss_val = float('inf')
            best_epoch = 0
            early_stopping_counter = 0

            train_loss_history = []
            val_loss_history = []

            for epoch in range(self.n_steps):
                start = 0
                end_train = start + batch_size
                end_val = start + batch_size_val
                batch_xs = X[start:end_train]
                batch_zs = Z[start:end_train]

                batch_xs_val = X_val[start:end_val]
                batch_zs_val = Z_val[start:end_val]

                train_cost = self.train_step(batch_xs, batch_zs, optimizer)
                val_cost = self.validation_step(batch_xs_val, batch_zs_val)

                train_loss_history.append(train_cost.numpy())
                val_loss_history.append(val_cost.numpy())

                if val_cost < best_loss_val:
                    best_loss_val = val_cost
                    best_epoch = epoch
                    best_weights = [tf.identity(self.W), tf.identity(self.b1),tf.identity(self.W2), tf.identity(self.b2) ]
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

                if early_stopping_counter > 10:  # early stopping criteria
                    break

            if best_loss_val < best_config["best_loss_val"]:
                best_config["n_hidden"] = n_hidden
                best_config["best_epoch"] = best_epoch
                best_config["best_loss_val"] = best_loss_val
                best_config["best_weights"] = best_weights
                best_config["train_loss_history"] = train_loss_history
                best_config["val_loss_history"] = val_loss_history

        # Set the model to the best configuration
        self.build_hidden_layer(input_dim, best_config["n_hidden"])
        self.W.assign(best_config["best_weights"][0])
        self.b1.assign(best_config["best_weights"][1])
        self.W2.assign(best_config["best_weights"][2])
        self.b2.assign(best_config["best_weights"][3])
        self.history = {
            'train_loss': best_config["train_loss_history"],
            'val_loss': best_config["val_loss_history"]
        }

        return best_config

    def transfer(self, X):
        """
        Reconstruct X in the target domain.
        """
        if X.ndim == 1:
            X = X[np.newaxis, :]

        return self.call(X)

    def plot_history(self):
        """
        Plot the training and validation loss history.
        """
        if not self.history['train_loss'] or not self.history['val_loss']:
            raise ValueError("No training history available to plot.")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

def fit_te(te, df_source_train, df_target_train,df_source_val, df_target_val, one_to_one):
    """
    Fit the ITE using the given source and target domain dataframes and bipartite strategy
    """
    inputs_train, outputs_train = [], []
    inputs_val, outputs_val = [], []

    user_class_pairs = [(u, u) for u in df_source_train['User_type'].unique()]

    if one_to_one:
        iter_fun = zip
    else:
        iter_fun = product

    for u1, u2 in user_class_pairs:
        idx1 = list(df_source_train[df_source_train['User_type']==u1].index)
        idx2 = list(df_target_train[df_target_train['User_type']==u2].index)
        idx = np.array(list(iter_fun(idx1, idx2)))
        inputs_train.append(df_source_train.loc[idx[:, 0]].values)
        outputs_train.append(df_target_train.loc[idx[:, 1]].values)

    inputs_train = np.concatenate(inputs_train)
    outputs_train = np.concatenate(outputs_train)
    
    for u1_val, u2_val in user_class_pairs:
        idx1_val = list(df_source_val[df_source_val['User_type']==u1_val].index)
        idx2_val = list(df_target_val[df_target_val['User_type']==u2_val].index)
        idx_val = np.array(list(iter_fun(idx1_val, idx2_val)))
        inputs_val.append(df_source_val.loc[idx_val[:,0]].values)
        outputs_val.append(df_target_val.loc[idx_val[:,1]].values)
    
    inputs_val = np.concatenate(inputs_val)
    outputs_val = np.concatenate(outputs_val)
    
#     best_layers, best_steps  = te.fit_layers(inputs_train, outputs_train,inputs_val,outputs_val)
    best_config  = te.fit_layers(inputs_train, outputs_train,inputs_val,outputs_val)
    
#     te.fit(inputs_train, outputs_train,inputs_val,outputs_val, best_layers, best_steps)
    print(f"Best configuration: {best_config}")
    return te

def normalize(df):
    scaler = MinMaxScaler()
    scaler.fit(df)
    scaled = scaler.transform(df)
    scaled_df = pd.DataFrame(scaled, columns=df.columns)
    
    return scaled_df, scaler

def split_dataset(df,train_ratio, val_ratio):
    sample_index = list(np.random.choice(df.index.get_level_values(0), int(round((len(df) * train_ratio), 0)), 
                                         replace=False))
    other_index = list(df.index.get_level_values(0).difference(sample_index))
    val_index = list(np.random.choice(other_index, int(round((len(other_index) * val_ratio), 0)), replace=False))
    union_indices = set(sample_index).union(val_index)
    test_index = list(df.index.get_level_values(0).difference(union_indices))
    train = df.loc[df.index.get_level_values(0).isin(sample_index)]
    val = df.loc[df.index.get_level_values(0).isin(val_index)]
    test = df.loc[df.index.get_level_values(0).isin(test_index)]
    
    return train, val, test

def jdrf(x_train, y_train,x_test, y_test):
#     param_grid = {'bootstrap': [False, True],
#               'max_depth': [30,40],
#               'min_samples_leaf': [2,4],
#               'min_samples_split': [5,6,7],
#               'n_estimators': [500, 1000],
#               'criterion':['gini']
#              }

#     cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#     grid_rf = GridSearchCV(RandomForestClassifier(), param_grid, refit = True,
#                            cv = cv_inner, n_jobs = 1, scoring = 'accuracy')
    
#     tuning = grid_rf.fit(x_train, y_train)

#     best_hyperparams = tuning.best_params_

#     best_model = RandomForestClassifier(random_state=42, **best_hyperparams)
    best_model = RandomForestClassifier(random_state=42)

    trained_model = best_model.fit(x_train, y_train)
    pred_prob = trained_model.predict_proba(x_test)
    pred = trained_model.predict(x_test)
    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    
    return acc, pre, rec, f1, pred_prob

def transfer_learning(user_id,source_data,target_data):
    set_seeds()
    BBMAS_TE_FACTORY = lambda: TransferEncoder(n_steps=1000)
    te_factory = BBMAS_TE_FACTORY
    te = te_factory()

    i = user_id
    src_user = source_data.loc[source_data.index.get_level_values(0)==i]
    src_user.insert(loc = len(src_user.columns),column = 'User_type',value = 1)
    src_user.index = src_user.index.map(lambda x: ','.join(map(str, x)))

    tar_user = target_data.loc[target_data.index.get_level_values(0)==i]
    tar_user.insert(loc = len(tar_user.columns),column = 'User_type',value = 1)
    col_order = src_user.columns
    tar_user = tar_user.reindex(columns=col_order)
    tar_user.index = tar_user.index.map(lambda x: ','.join(map(str, x)))

    src_all_imposters = source_data.loc[source_data.index.get_level_values(0)!=i]
    src_all_imposters = src_all_imposters[common_features]
    src_all_imposters.insert(loc = len(src_all_imposters.columns),column = 'User_type',value = 0)
    src_all_imposters.index = src_all_imposters.index.map(lambda x: ','.join(map(str, x)))
    src_imposter_sample_indices = list(np.random.choice(src_all_imposters.index.get_level_values(0), size=len(src_user), 
                                                  replace=False))
    src_imposters = src_all_imposters.loc[src_all_imposters.index.get_level_values(0).isin(src_imposter_sample_indices)]

    tar_all_imposters = target_data.loc[target_data.index.get_level_values(0)!=i]
    tar_all_imposters = tar_all_imposters[common_features]
    tar_all_imposters.insert(loc = len(tar_all_imposters.columns),column = 'User_type',value = 0)
    tar_all_imposters = tar_all_imposters.reindex(columns=col_order)
    tar_all_imposters.index = tar_all_imposters.index.map(lambda x: ','.join(map(str, x)))
    tar_imposter_sample_index = list(np.random.choice(tar_all_imposters.index.get_level_values(0), size=len(tar_user), 
                                                  replace=False))
    tar_imposters = tar_all_imposters.loc[tar_all_imposters.index.get_level_values(0).isin(tar_imposter_sample_index)]

    src_train_gen, src_val_gen, src_test_gen = split_dataset(src_user, 0.3, 0.2)
    tar_train_gen, tar_val_gen, tar_test_gen = split_dataset(tar_user, 0.3, 0.2)

    src_train_imp, src_val_imp, src_test_imp = split_dataset(src_imposters, 0.3, 0.2)
    tar_train_imp, tar_val_imp, tar_test_imp = split_dataset(tar_imposters, 0.3, 0.2)

    src_train = pd.concat([src_train_gen,src_train_imp])
    src_val = pd.concat([src_val_gen,src_val_imp])
    src_test = pd.concat([src_test_gen,src_test_imp])

    tar_train = pd.concat([tar_train_gen,tar_train_imp])
    tar_val = pd.concat([tar_val_gen,tar_val_imp])
    tar_test = pd.concat([tar_test_gen,tar_test_imp])

    X_src_train = src_train.iloc[:,:-1]
    Y_src_train = src_train.iloc[:,-1]
    X_src_val = src_val.iloc[:,:-1]
    Y_src_val = src_val.iloc[:,-1]
    X_src_test = src_test.iloc[:,:-1]
    Y_src_test = src_test.iloc[:,-1]

    indexes_src_tarin = X_src_train.index
    X_src_train, src_scaler = normalize(X_src_train)
    X_src_train.index = indexes_src_tarin
    indexes_src_val = X_src_val.index
    val_scale = src_scaler.transform(X_src_val)
    X_src_val = pd.DataFrame(val_scale, columns=X_src_val.columns)
    X_src_val.index = indexes_src_val
    
    indexes_src_test = X_src_test.index
    test_scale = src_scaler.transform(X_src_test)
    X_src_test = pd.DataFrame(test_scale, columns=X_src_test.columns)
    X_src_test.index = indexes_src_test

    src_train_new = pd.concat([X_src_train,Y_src_train], axis=1)
    src_val_new = pd.concat([X_src_val,Y_src_val], axis=1)
    src_test_new = pd.concat([X_src_test,Y_src_test], axis=1)
    
    X_tar_train = tar_train.iloc[:,:-1]
    Y_tar_train = tar_train.iloc[:,-1]
    X_tar_val = tar_val.iloc[:,:-1]
    Y_tar_val = tar_val.iloc[:,-1]
    X_tar_test = tar_test.iloc[:,:-1]
    Y_tar_test = tar_test.iloc[:,-1]

    indexes_tar_tarin = X_tar_train.index
    X_tar_train, tar_scaler = normalize(X_tar_train)
    X_tar_train.index = indexes_tar_tarin
    indexes_tar_val = X_tar_val.index
    val_scale = tar_scaler.transform(X_tar_val)
    X_tar_val = pd.DataFrame(val_scale, columns=X_tar_val.columns)
    X_tar_val.index = indexes_tar_val
    
    indexes_tar_test = X_tar_test.index
    test_tar_scale = tar_scaler.transform(X_tar_test)
    X_tar_test = pd.DataFrame(test_tar_scale, columns=X_tar_test.columns)
    X_tar_test.index = indexes_tar_test

    tar_train_new = pd.concat([X_tar_train,Y_tar_train], axis=1)
    tar_val_new = pd.concat([X_tar_val,Y_tar_val], axis=1)
    tar_test_new = pd.concat([X_tar_test,Y_tar_test], axis=1)
    
    te = fit_te(te, src_train_new, tar_train_new, src_val_new,tar_val_new, one_to_one=False)
    
    te.plot_history()

    src_test_new = src_test_new.astype('float32')
    
    src_test_new.values[:] = te.transfer(src_test_new.values)
    src_test_new['User_type'] = Y_src_test
    
    return src_test_new, tar_train_new, tar_val_new, tar_test_new

def user_training(transform_set, tar_train, tar_val, tar_test):
    
    train_set_target =  pd.concat([tar_train, tar_val,transform_set], ignore_index=True)
    
    train_x = train_set_target.iloc[:,:-1]
    tarin_y = train_set_target.iloc[:,-1]
    
    test_set_target = tar_test

    test_x = test_set_target.iloc[:,:-1]
    test_y = test_set_target.iloc[:,-1]

    acc, pre, rec, f1, pred_prob = jdrf(train_x,tarin_y,test_x, test_y)
    
    return acc, pre, rec, f1, pred_prob, test_y

def calculating_eer(pred_prob, true_label):
    fpr, tpr, _ = roc_curve(true_label,pred_prob)
    fnr = [1-x for x in tpr]
    diff_values = [abs(fnr-fpr) for fnr, fpr in zip (fnr, fpr)]
    min_diff_index = diff_values.index(min(diff_values))
    eer_fnr = fnr[min_diff_index]
    eer_fpr = fpr[min_diff_index]
    eer = (eer_fnr + eer_fpr)/2
    
    return eer


source_data = pd.read_csv('C:/Users/s3929438/all_features_mobile_100_all_final_latest.csv', index_col=[1,2])
target_data = pd.read_csv('C:/Users/s3929438/all_features_tablet_100_all_latest.csv',index_col=[1,2])
# all_features_tablet_100_all_latest

source_data = source_data.loc[:, ~source_data.columns.str.contains('^Unnamed')]
target_data = target_data.loc[:, ~target_data.columns.str.contains('^Unnamed')]

source_data = source_data.astype('float32')
target_data = target_data.astype('float32')

common_features = ['mean_F1', 'mean_F2', 'mean_F3', 'mean_F4', 'Tri_graph', 'Error_rate_%','mean_hold_time',
                    'mean_F1_dis_1_LL','mean_F1_dis_2_LL', 'mean_F1_dis_2_RR', 'mean_F1_dis_3_LL','mean_F1_dis_3_RR', 
                    'mean_F2_dis_1_LL','mean_F2_dis_2_LL', 'mean_F2_dis_2_RR', 'mean_F2_dis_3_LL','mean_F2_dis_3_RR', 
                    'mean_F3_dis_2_RR', 'mean_F3_dis_3_LL','mean_F4_dis_1_LL','mean_F4_dis_2_LL', 'mean_F4_dis_2_RR',
                    'mean_F4_dis_3_LL']


source_data = source_data[common_features]
target_data = target_data[common_features]

source_new, train, val, test = transfer_learning(1,source_data, target_data)
print(source_new)
acc, pre, rec, f1, pred_prob, test_y = user_training(source_new,train, val, test)
eer = calculating_eer (pred_prob[:,-1].tolist(),test_y.tolist())
print(eer)