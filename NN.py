


def NN(data_sets, y_train, y_test, early_stop_patience=2, filename='train', maxlen=50):

    results, time = f.init_log_for_training()

    batch_size = 20
    for data_set in data_sets:
        labels = data_set[0]

        Train = data_set[1]
        Test = data_set[2]

        list_train = []
        list_test = []

        for i in Train:
            list_train.append(i[0])
        for i in Test:
            list_test.append(i[0])

        x_val = list_train[:1000]
        X_train = list_train[1000:]
        y_val = y_train[:1000]
        Y_train = y_train[1000:]

        model = "https://tfhub.dev/google/nnlm-en-dim50/2"
        hub_layer = hub.KerasLayer(model, input_shape=[], output_shape=[],dtype=tf.string, trainable=True)

        hub_layer(list_train[:10])

        model = tf.keras.Sequential()
        model.add(hub_layer)
        model.add(tf.keras.layers.Dense(160, activation='relu'))
        #model.add(tf.keras.layers.Dense(100, activation = 'tanh'))
        model.add(tf.keras.layers.Dense(1)) ##outputs the log-odds of the true class
        model.summary()


        model.compile(optimizer='adam',
                loss=tf.losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        history = model.fit(X_train, 
                    Y_train, 
                    epochs = 5,
                    batch_size = 20,
                    validation_data = (x_val, y_val), 
                    verbose = 1)
        max_value = max(history.history['val_accuracy'])
        max_index = history.history['val_accuracy'].index(max_value)

        ratio_train, ratio_test = f.lenghts_catch(data_set[1], data_set[2], lenght_of_sentence = maxlen)
        results = model.evaluate(, test_y_train) 
        new_row = {"Running ID": time, 
               "Model Name": 'NN', 
               "Expand Contractions" : labels[0], 
               "Basic Preprocessing": labels[1],
               "Grammar Correction":labels[2],
               "Simplify Negations":labels[3],
               "Lemmatize":labels[4],
               "Remove Stop Words": labels[5],
               "No. of Sentences": len(data_set[1]),
               "Train Accuracy STOP": history.history['accuracy'][max_index],
               "Test Accuracy STOP": history.history['val_accuracy'][max_index],
               "Train Loss STOP": history.history['loss'][max_index],
               "Test Loss STOP": history.history['val_loss'][max_index],
                  "Train_sentence_fully_catched_ratio": ratio_train,
                   "Test_sentence_fully_catched_ratio": ratio_test,
                   "Length of Sentence":maxlen,
                  "Batch size of RNN":batch_size,
                  "label_array": str(labels)
                  }
        results = results.append(new_row, ignore_index=True)
      
        try:
            results.to_csv("results/"+filename+"_results_"+time+".csv")
        except: 
            continue
            
      # CLEANING
        del labels
        del Train
        del Test
        del list_train
        del list_test
        del history
        del model 