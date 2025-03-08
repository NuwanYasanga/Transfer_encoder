import pandas as pd
from itertools import product
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from statistics import mean, stdev
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


class TransferEncoder(tf.keras.Model):
    set_seeds()
    def __init__(self, n_steps, epochs):
        super(TransferEncoder, self).__init__()
        self.n_steps = n_steps
        self.n_epochs = epochs
        self.hidden_layer = None
        self.output_layer = None
        self.initial_weights = None
        self.final_weights = None
        
    def build_hidden_layer(self, input_dim, n_hidden):
 

        self.hidden_layer=layers.Dense(n_hidden,
                                         kernel_initializer=tf.keras.initializers.RandomUniform(
                                             minval=-1.0 / np.sqrt(input_dim),
                                             maxval=1.0 / np.sqrt(input_dim),
                                             seed=1234),
                                         bias_initializer='zeros')
    
        
        self.dropout = layers.Dropout(rate=0.5)
        
        self.output_layer = layers.Dense(input_dim,
            kernel_initializer=lambda shape, dtype=None: tf.transpose(self.hidden_layer.kernel),
            bias_initializer='zeros')
        
        self.hidden_layer.trainable = True
        self.output_layer.trainable = True
        self.hidden_layer.build((None, input_dim))
        self.output_layer.build((None, n_hidden))
        
        self.initial_weights = self.get_weights()
        
    @tf.function
    def call(self, x, training=False):
        y = tf.nn.tanh(self.hidden_layer(x))
        y = self.dropout(y, training=training)
        z_hat = tf.nn.tanh(self.output_layer(y))
        return z_hat
    
    @tf.function
    def train_step(self, data):
        x_batch, z_batch = data
        with tf.GradientTape() as tape:
            z_hat = self(x_batch, training=True)
            cost = tf.reduce_mean(tf.keras.losses.mean_squared_error(z_batch, z_hat))
            l2_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in self.trainable_weights])
            cost += 0.01 * l2_loss
        gradients = tape.gradient(cost, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        return cost
    

    def validation_step(self, data):
        x_batch, z_batch = data
        z_hat = self(x_batch, training=False)
        cost = tf.reduce_mean(tf.keras.losses.mean_squared_error(z_batch, z_hat))
        l2_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in self.trainable_weights])
        cost += 0.01 * l2_loss
        return cost
    
#     def create_batches(self, X, Z, batch_size):
#         dataset = tf.data.Dataset.from_tensor_slices((X, Z))
#         dataset = dataset.shuffle(buffer_size=len(X)).batch(batch_size)
#         return dataset


    def fit_layers(self, X_train, Z_train, X_val, Z_val, batch_size_train, batch_size_val):
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
            self.build_hidden_layer(input_dim,n_hidden)
            self.compile(optimizer =tf.keras.optimizers.Adam(learning_rate=0.001), 
                         loss = tf.keras.losses.MeanSquaredError(reduction="sum_over_batch_size"))
            
            best_loss_val = float('inf')
            best_epoch = 0
            early_stopping_counter = 0

            train_loss_history = []
            val_loss_history = []

            for epoch in range(self.n_epochs):
                epoch_train_loss = 0
                epoch_val_loss = 0
                num_batches = 0
                    
                total_batches_train = int(X_train.shape[0]/batch_size_train)
                for start_idx in range(0, total_batches_train, batch_size_train):
                    end_idx = min(start_idx + batch_size_train, total_batches_train)
                    x_batch = X_train[start_idx:end_idx]
                    z_batch = Z_train[start_idx:end_idx]

               
                    train_cost = self.train_step((x_batch, z_batch))
                    epoch_train_loss += train_cost
                    num_batches += 1

                epoch_train_loss /= num_batches
                train_loss_history.append(epoch_train_loss.numpy())
                

                num_batches = 0
                total_batches_val = int(X_val.shape[0]/batch_size_val)
                for start_idx in range(0, total_batches_val, batch_size_val):
                    end_idx = min(start_idx + batch_size_val, total_batches_val)
                    x_batch_val = X_train[start_idx:end_idx]
                    z_batch_val = Z_train[start_idx:end_idx]
                    val_cost = self.validation_step((x_batch_val, z_batch_val))
                    epoch_val_loss += val_cost
                    num_batches += 1

                epoch_val_loss /= num_batches
                val_loss_history.append(epoch_val_loss.numpy())
                

                if epoch_val_loss < best_loss_val:
                    best_loss_val = epoch_val_loss
                    best_epoch = epoch
                    best_weights = self.get_weights()
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

                if early_stopping_counter > 10:
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
        self.set_weights(best_config["best_weights"])
        self.history = {
            'train_loss': best_config["train_loss_history"],
            'val_loss': best_config["val_loss_history"]
        }
        
        self.final_weights = best_config["best_weights"]

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
        
    def get_initial_final_weights(self):
        return self.initial_weights, self.final_weights

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

    
    best_config  = te.fit_layers(inputs_train, outputs_train,inputs_val,outputs_val, len(idx1) * 2, len(idx2_val) *2)
    
#     te.fit(inputs_train, outputs_train,inputs_val,outputs_val, best_layers, best_steps)
#     print(f"Best configuration: {best_config}")
    
    return te

def normalize(df):
    scaler = MinMaxScaler()
    scaler.fit(df)
    scaled = scaler.transform(df)
    scaled_df = pd.DataFrame(scaled, columns=df.columns)
    
    return scaled_df, scaler

def split_dataset(df,train_ratio, val_ratio):
    np.random.seed(42)
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

def user_training(transform_set, tar_train, tar_val, tar_test):
    
    train_set_target =  pd.concat([tar_train, tar_val, transform_set], ignore_index=True)
    
    train_x = train_set_target.iloc[:,:-1]
    tarin_y = train_set_target.iloc[:,-1]
    
    test_set_target = tar_test

    test_x = test_set_target.iloc[:,:-1]
    test_y = test_set_target.iloc[:,-1]
    
#     indexes_tar_tarin = train_x.index
#     train_x, tar_scaler = normalize(train_x)
#     train_x.index = indexes_tar_tarin
    
#     indexes_tar_test = test_x.index
#     test_scale = tar_scaler.transform(test_x)
#     test_x = pd.DataFrame(test_scale, columns=test_x.columns)
#     test_x.index = indexes_tar_test

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

def jdrf(x_train, y_train,x_test, y_test):
    best_model = RandomForestClassifier(random_state=42)

    trained_model = best_model.fit(x_train, y_train)
    pred_prob = trained_model.predict_proba(x_test)
    pred = trained_model.predict(x_test)
    predict_list = pred_prob[:,-1].tolist()
    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    
    return acc, pre, rec, f1, predict_list


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

set_seeds()
BBMAS_TE_FACTORY = lambda: TransferEncoder(n_steps=1000, epochs=500)
te_factory = BBMAS_TE_FACTORY
te = te_factory()

i = 1
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
    
src_test_new = src_test_new.astype('float32')
src_test_new.values[:] = te.transfer(src_test_new.values)
src_test_new['User_type'] = Y_src_test


acc, pre, rec, f1, pred_prob, test_y = user_training(src_test_new,tar_train_new, tar_val_new, tar_test_new)

true_label = test_y.tolist()
user_id_list =  [i] * len(pred_prob)

eer = calculating_eer(pred_prob,true_label)

df_final = pd.DataFrame([[i, acc, pre, rec, f1, eer]], columns = ['User', 'Accuracy', 'Precision','Recall','F1', 'EER'])
df_final1 = pd.DataFrame(list(zip(user_id_list,pred_prob, true_label)),columns = ['User', 'Predicted_probability', 'True Label'])

print(df_final)

#filename = f'Data_output/Transfer_encoder_edited_performance/User_' + str(user_id) +'.csv'
#df_final.to_csv(filename)

#filename1 = f'Data_output/Transfer_encoder_edited_probs/User_' + str(user_id) +'.csv'
#df_final1.to_csv(filename1)


