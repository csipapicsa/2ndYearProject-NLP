###

import imports as ii
import functions as f
import preprocessing as pp
import neuralnetworks as nn

def trainRNN(data_sets, y_train, y_test, early_stop_patience=2, filename="train"):
    ### INIT RESULTS
    results, time = f.init_log_for_training()
    # Sentencies max length:

    max_len = 40 # maximum length of sentencies
    # its 2 FOR NORMAL CASE
    batch_size = 25 # bigger = faster train but super unaccurate, lower = slower. Its usually 50

    for data_set in data_sets:
        # Tokenizer
        labels = data_set[0]
        # check whenever combination is already checked. Working only with RNN!:
        print("Combinations: ", labels)

        tokenizer = pp.tokenizer_init(data_set[1], data_set[2])
        Train = tokenizer.texts_to_sequences(data_set[1])
        Test = tokenizer.texts_to_sequences(data_set[2])
            # Sequencer 
        X_train_p = pp.sequence_pad(Train, maxlen=max_len) # there are several attributes which can be defined, basic = first 50 words 
        X_test_p = pp.sequence_pad(Test,maxlen=max_len)

            # TRAIN
        #print("shapes: ", X_train_p.shape, X_test_p.shape)
        history, model = nn.RNN_train(X_train_p, y_train, X_test_p, y_test, tokenizer, 
                                      maxlen=max_len, early_stop_patience=early_stop_patience, batch_size=batch_size)
        ### LOGGING INIT - RNN
        # GET the index of the highest test ACCURACY where the RNN model stopped to TRAIN
        max_value = max(history.history['val_accuracy'])
        max_index = history.history['val_accuracy'].index(max_value)
        # How many percent of the sentencies were catched full
        ratio_train, ratio_test = f.lengths_catch(data_set[1], data_set[2], length_of_sentence=max_len)

        new_row = {'Running ID':time, 
               "Model Name":"RNN", 
              "Expand Contractions":labels[0],
              "Basic Preprocessing":labels[1],
              "Grammar Correction":labels[2],
               "Simplify Negotiations": labels[3],
              "Lemmatize": labels[4],
              "Remove Stop Words": labels[5],
              "No. of Sentences": len(data_set[1]),
              "Train Accuracy STOP": history.history['accuracy'][max_index],
              "Test Accuracy STOP": history.history['val_accuracy'][max_index],
              "Train Loss STOP": history.history['loss'][max_index],
              "Test Loss STOP": history.history['val_loss'][max_index],
                  "Train_sentence_fully_catched_ratio": ratio_train,
                   "Test_sentence_fully_catched_ratio": ratio_test,
                   "Length of Sentence":max_len,
                  "Batch size of RNN":batch_size}

        results = results.append(new_row, ignore_index=True)
        # maybe we dont need it in every round but how knows
        try:
            results.to_csv("results/"+filename+"_results_"+time+".csv")
        except: 
            continue
        f.plot_model_history(history)    
        # CLEAN
        del labels
        del tokenizer
        del Train
        del Test
        del X_train_p
        del X_test_p
        del history
        del model 

    # save results again
    print("Combinations were checked")
    try:
        results.to_csv("results/"+filename+"_results_"+time+".csv")
    except:
        pass
        
    return None