# Adapted from: https://github.com/aws-samples/amazon-sagemaker-script-mode/blob/master/tf-2-workflow-smpipelines/train_model/train.py
import argparse
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Reshape, Normalization, Flatten, Dropout 
from tensorflow.keras.layers import Concatenate, GaussianNoise, Lambda
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, RNN, GRU

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--units', type=int, default=8)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--l2_regularization', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))

    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    return parser.parse_known_args()

def get_normalization_statistics(train_dir):

    global_mean = np.load(os.path.join(train_dir, 'global_mean.npy'))
    global_stddev = np.load(os.path.join(train_dir, 'global_stddev.npy'))
    print('global_mean', global_mean,'global_stddev', global_stddev)

    return global_mean, global_stddev

def get_train_data(train_dir):

    continuous_train_inputs = np.load(os.path.join(train_dir, 'continuous_train_inputs.npy'))
    categorical_train_inputs = np.load(os.path.join(train_dir, 'categorical_train_inputs.npy'))
    train_targets = np.load(os.path.join(train_dir, 'train_targets.npy'))
    print(
        'continuous_train_inputs:', 
        continuous_train_inputs.shape, 
        'categorical_train_inputs:', 
        categorical_train_inputs.shape,
        'train_targets:',
        train_targets.shape
    )

    return continuous_train_inputs, categorical_train_inputs, train_targets


def get_test_data(test_dir):

    continuous_test_inputs = np.load(os.path.join(test_dir, 'continuous_test_inputs.npy'))
    categorical_test_inputs = np.load(os.path.join(test_dir, 'categorical_test_inputs.npy'))
    test_targets = np.load(os.path.join(test_dir, 'test_targets.npy'))
    print(
        'continuous_test_inputs:', 
        continuous_test_inputs.shape, 
        'categorical_test_inputs:', 
        categorical_test_inputs.shape,
        'test_targets:',
        test_targets.shape
    )

    return continuous_test_inputs, categorical_test_inputs, test_targets

def get_val_data(val_dir):

    continuous_val_inputs = np.load(os.path.join(val_dir, 'continuous_val_inputs.npy'))
    categorical_val_inputs = np.load(os.path.join(val_dir, 'categorical_val_inputs.npy'))
    val_targets = np.load(os.path.join(val_dir, 'val_targets.npy'))
    print(
        'continuous_val_inputs:', 
        continuous_val_inputs.shape, 
        'categorical_val_inputs:', 
        categorical_val_inputs.shape,
        'val_targets:',
        val_targets.shape
    )

    return continuous_val_inputs, categorical_val_inputs, val_targets

def create_model(norm_mean, norm_stddev, units, noise, l2_regularization, dropout,
                continuous_input_shape, categorical_input_shape, target_shape):
    continuous_input = Input(
        shape=continuous_input_shape, 
        name="continuous_input"
    )

    categorical_input = Input(
        shape=categorical_input_shape, 
        name="categorical_input"
    )

    global_mean_tf = tf.constant(norm_mean, dtype=tf.float32)
    global_stddev_tf = tf.constant(norm_stddev, dtype=tf.float32)
    target_mean_tf = tf.constant(norm_mean[0], dtype=tf.float32)
    target_stddev_tf = tf.constant(norm_stddev[0], dtype=tf.float32)

    normalized_input = Lambda(lambda x: (x - global_mean_tf) / global_stddev_tf)(continuous_input)
    noisy_input = GaussianNoise(stddev=noise)(normalized_input)
    combined_inputs = Concatenate(axis=-1)([noisy_input, categorical_input])
    reshaped = Reshape(target_shape=(-1, combined_inputs.shape[3]))(combined_inputs)
    gru = GRU(units, kernel_regularizer=tf.keras.regularizers.l2(l2_regularization))(reshaped)
    dropout = Dropout(dropout)(gru)
    dense = Dense(target_shape[0]*target_shape[1])(dropout)
    outputs = Reshape(target_shape=target_shape)(dense)
    denormalized_outputs = Lambda(lambda x: x * target_stddev_tf + target_mean_tf)(outputs)

    return Model(inputs=[continuous_input, categorical_input], outputs=denormalized_outputs)
    
if __name__ == "__main__":

    args, _ = parse_args()
 
    print('Training data location: {}'.format(args.train))
    print('Test data location: {}'.format(args.test))
    print('Validation data location: {}'.format(args.validation))
    continuous_train_inputs, categorical_train_inputs, train_targets = get_train_data(args.train)
    continuous_test_inputs, categorical_test_inputs, test_targets = get_test_data(args.test)
    continuous_val_inputs, categorical_val_inputs, val_targets = get_val_data(args.validation)
    
    global_mean, global_stddev = get_normalization_statistics(args.train)
    global_mean_list = global_mean.tolist()
    global_stddev_list = global_stddev.tolist()

    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    units = args.units
    noise = args.noise
    l2_regularization = args.l2_regularization
    dropout = args.dropout
    print('batch_size = {}, epochs = {}, learning rate = {}, units = {}, noise = {}, l2_regularization = {}, dropout = {}'
          .format(batch_size, epochs, learning_rate, units, noise, l2_regularization, dropout))
    
    
    model = create_model(global_mean_list, global_stddev_list, units, noise, l2_regularization, dropout,
                        continuous_input_shape=continuous_train_inputs.shape[1:],
                        categorical_input_shape=categorical_train_inputs.shape[1:],
                        target_shape=train_targets.shape[1:]
    )
    
    model.compile(loss='mean_squared_error', metrics=['mean_absolute_error'], optimizer=tf.keras.optimizers.Adam(learning_rate))

    print(model.summary())
    
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=10000,
        patience=6,
        verbose=0,
        restore_best_weights=True
    )


    history = model.fit(
        [continuous_train_inputs, categorical_train_inputs], 
        train_targets, 
        validation_data=([continuous_test_inputs, categorical_test_inputs], test_targets), 
        epochs=epochs, 
        batch_size=batch_size, 
        callbacks=[es_callback],
        verbose=2
    )

    # plt.plot(history.history['loss'], label='Train Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    eval = model.evaluate([continuous_val_inputs, categorical_val_inputs], val_targets, verbose=2)
    print(f"\n Test MSE: {eval[0]} Test MAE: {eval[1]}")
    
    model.save(args.sm_model_dir + '/1')