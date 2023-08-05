# Load Libs
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, f1_score, accuracy_score

import json

# Load preprocessing script and its vars 
from preprocessing import *

n_to_test = []
accuracy = []
f1_micro = []
f1_macro = []
labeld_dps = []

def test_al(labeld_x, labeld_y, unlabeld_x, unlabeld_y, test_x, test_y, dps_to_label):
    
    print("Starting sampling")
    
    results = []
    # iteration
    while len(labeld_x) < 1500:
        
        print(len(labeld_x))
        
        # Create a mask for reverse selection -> Everything but the dps_to_label
        mask_reverse_select = np.ones(len(unlabeld_x), dtype=bool)
        mask_reverse_select[dps_to_label] = 0
        
        # Add the data and the labels -> simulated learning
        labeld_x = np.concatenate((labeld_x, unlabeld_x[dps_to_label]))
        labeld_y = np.concatenate((labeld_y, unlabeld_y[dps_to_label]))
        
        labeld_dps.append(dps_to_label.tolist())
        
        # Store the rest of the data 
        unlabeld_x = unlabeld_x[mask_reverse_select]
        unlabeld_y = unlabeld_y[mask_reverse_select]

        # Initialize the Multilabel SVM classifier
        weights = {0:1.0, 1:100.0}
        svc = SVC(kernel="rbf", class_weight=weights, probability=True)
            
        multilabel_classifier_svc = MultiOutputClassifier(svc, n_jobs=-1)
        
        # We can only create labels if we have positive and negative examples, therefore we have to check which labels are usable
        valid_labels = [i for i in range(labeld_y.shape[1]) if len(np.unique(labeld_y[:,i])) > 1]

        # Select a subset with only the valid labels
        labeld_y_valid = labeld_y[:,valid_labels]
        
        # Fit the classifier with the valid labels
        multilabel_classifier_svc = multilabel_classifier_svc.fit(
            labeld_x,
            labeld_y_valid)
        
        # Compute predictions on the test data for the valid labels 
        test_y_valid_predicted = multilabel_classifier_svc.predict(test_x)

        # For Performance Evaluation we however need predidictions for all labels
        # Due to class imbalance, we simply predict 0 for all labels as a frist guess
        test_y_predicted = np.zeros((test_y_valid_predicted.shape[0], unlabeld_y.shape[1]))

        # For our valid labels, we can update the dataframe with actual predictions
        test_y_predicted[:,valid_labels] = test_y_valid_predicted
        
        # We can now compute all performance metrics 
        n_to_test.append(len(labeld_x))
        accuracy.append(accuracy_score(test_y, test_y_predicted))
        f1_micro.append(f1_score(test_y, test_y_predicted, average="micro"))
        f1_macro.append(f1_score(test_y, test_y_predicted, average="macro"))
        
        # For our next iteration we need to find the datapoints about wich our classifier is most uncertain
        unlabeld_x_predproba = multilabel_classifier_svc.predict_proba(unlabeld_x)
        
        # Compute Entropy per datapoint 
        entropy_per_datapoint = np.zeros((unlabeld_x_predproba[0].shape[0]))
        for var in unlabeld_x_predproba:
            var_entropy = var[:,1]*np.log(var[:,1])
            entropy_per_datapoint = entropy_per_datapoint - var_entropy
        
        # Select 25 dps with highest Entropy
        dps_to_label = np.array([], dtype="int32")
        dps_to_label = np.append(dps_to_label, np.argsort(entropy_per_datapoint)[::-1][:25])
    
    
    print("Iteration complete")
    
    return results

test_al(
    np.empty((0,embeddings_train.shape[1]), dtype = embeddings_train.dtype),
    np.empty((0,labels_onehot_train.shape[1]), dtype = labels_onehot_train.dtype),
    embeddings_train,
    labels_onehot_train,
    embeddings_test, 
    labels_onehot_test,
    np.random.choice(len(embeddings_train), 25, replace=False))

results = {
    "n":n_to_test,
    "accuracy":accuracy,
    "f1_micro":f1_micro,
    "f1_macro":f1_macro,
    "labeld_dps":labeld_dps}

# Serializing json
json_object = json.dumps(results, indent=4)

# Writing to sample.json
with open("results/uncert_svm_embeddings_2.json", "w") as outfile:
    outfile.write(json_object)