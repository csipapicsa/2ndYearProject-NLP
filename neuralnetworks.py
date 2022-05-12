# time
from datetime import datetime

def RNN_train(X_train_p, y_train, X_test, y_test, tokenizer, maxlen=50):
    import numpy as np
    # convert the sets into a numpy array
    X_train_p = np.array(X_train_p)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    vocabulary_size = len(tokenizer.word_counts)
    
    from keras import Sequential
    from keras.layers import Embedding, LSTM, Dense, Dropout
    # get time if we want to saving the model
    dateTimeObj = datetime.now()
    save_time = str(dateTimeObj.year)+'-'+str(dateTimeObj.month)+'-'+str(dateTimeObj.day)+'-'+str(dateTimeObj.hour)+'-'+str(dateTimeObj.minute)+'-'+str(dateTimeObj.second)
    
    
    # define the model

    embedding_size=200 # bigger = slower train
    model=Sequential()
    model.add(Embedding(vocabulary_size+1, embedding_size, input_length=maxlen))
    model.add(LSTM(100))
    model.add(Dense(1, activation='tanh')) # shape of the labels, if its 2, than the y_ labels has a x* 2 shape
    
    model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

    #print(model.summary())
    
    # early stopping

    from keras.callbacks import ReduceLROnPlateau
    from keras.callbacks import EarlyStopping
    from keras.callbacks import ModelCheckpoint
    earlyStopping = EarlyStopping(monitor='val_accuracy', patience=2, verbose=1, mode='max',restore_best_weights=False)
    
    # activete this line, if you wanna save the best model: 
    # mcp_save = ModelCheckpoint('model/'+save_time+'-model.mdl_wts.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')
    
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=1, verbose=1, mode='min') 
    
    # patience: 10% of number of epochs. Anyway, it is just for stopping the validation, since we have model checkpoint its doesnt matter
    batch_size = 50 # lower = slower train, higher = faster train
    num_epochs = 10

    history = model.fit(X_train_p, y_train, validation_data=(X_test, y_test), 
                    batch_size=batch_size, 
                    epochs=num_epochs,
                    # callbacks=[earlyStopping, mcp_save, reduce_lr_loss]) # use this if you wanna save the model
                    callbacks=[earlyStopping, reduce_lr_loss])# for regularization)
    
    
    
    return history, model