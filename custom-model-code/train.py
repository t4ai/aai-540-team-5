# Define a script that trains the custom model in AWS script mode
# Adapted from: https://github.com/aws-samples/amazon-sagemaker-script-mode/blob/master/tf-2-workflow-smpipelines/train_model/train.py
import argparse
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Reshape, Normalization, Flatten, Dropout
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D

# Define a function to parse the model arguments
def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--l2_regularization', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))

    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    return parser.parse_known_args()


# Define a function to load the training data
def get_train_data(train_dir):
    
    train_inputs = np.load(os.path.join(train_dir, 'train_inputs.npy'))
    train_targets = np.load(os.path.join(train_dir, 'train_targets.npy'))
    
    print('train_inputs:', train_inputs.shape, 'train_targets:', train_targets.shape)

    return train_inputs, train_targets


# Define a function to load the test data
def get_test_data(test_dir):
    
    test_inputs = np.load(os.path.join(test_dir, 'test_inputs.npy'))
    test_targets = np.load(os.path.join(test_dir, 'test_targets.npy'))
    
    print('test_inputs:', test_inputs.shape, 'test_targets:', test_targets.shape)

    return test_inputs, test_targets


# Define a function to load the validation data
def get_val_data(val_dir):
    
    val_inputs = np.load(os.path.join(val_dir, 'val_inputs.npy'))
    val_targets = np.load(os.path.join(val_dir, 'val_targets.npy'))
    
    print('val_inputs:', val_inputs.shape, 'val_targets:', val_targets.shape)

    return val_inputs, val_targets


# Define the model architecture
def create_model(l2_regularization, dropout, input_shape, target_shape):
    
    input_layer = Input(shape=input_shape)
    reshaped_input = Reshape(target_shape=(-1, input_shape[2]))(input_layer)
    conv = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(reshaped_input)
    pooled = MaxPooling1D(pool_size=2)(conv)
    dropout1 = Dropout(dropout)(pooled)
    gru = GRU(8, kernel_regularizer=tf.keras.regularizers.l2(l2_regularization))(dropout1)
    dropout2 = Dropout(dropout)(gru)
    dense = Dense(target_shape[0]*target_shape[1])(dropout2)
    outputs = Reshape(target_shape=target_shape)(dense)
    

    return Model(inputs=input_layer, outputs=outputs)
    
if __name__ == "__main__":

    # Parse arguments
    args, _ = parse_args()
 
    # Load the data
    print('Training data location: {}'.format(args.train))
    print('Test data location: {}'.format(args.test))
    print('Validation data location: {}'.format(args.validation))
    train_inputs, train_targets = get_train_data(args.train)
    test_inputs, test_targets = get_test_data(args.test)
    val_inputs, val_targets = get_val_data(args.validation)

    # Set the model training hyperparameters
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    l2_regularization = args.l2_regularization
    dropout = args.dropout
    print('batch_size = {}, epochs = {}, learning rate = {}, l2_regularization = {}, dropout = {}'
          .format(batch_size, epochs, learning_rate, l2_regularization, dropout))
    
    # Instantiate the model
    model = create_model(
        l2_regularization, 
        dropout, 
        input_shape=train_inputs.shape[1:], 
        target_shape=train_targets.shape[1:]
    )
    
    # Compile the model
    model.compile(
        loss='mean_squared_error', 
        metrics=[tf.keras.metrics.RootMeanSquaredError(), 'mean_absolute_error'], 
        optimizer=tf.keras.optimizers.Adam(learning_rate)
    )

    print(model.summary())
    
    # Define an early stopping callback for the model
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=4,
        verbose=1,
        restore_best_weights=True
    )


    # Train the model
    history = model.fit(
        train_inputs, 
        train_targets, 
        validation_data=(test_inputs, test_targets), 
        epochs=epochs, 
        batch_size=batch_size, 
        callbacks=[es_callback],
        verbose=2
    )


    # Evaluate the model on the validation set
    eval = model.evaluate(val_inputs, val_targets, verbose=2)
    print(f"\n Test MSE: {eval[0]} Test RMSE: {eval[1]} Test MAE: {eval[2]}")
    
    # Save the model
    model.save(args.sm_model_dir + '/1')