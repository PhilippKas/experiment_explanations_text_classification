# Load Libs
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, f1_score, accuracy_score

import json

# Load preprocessing script and its vars 
from preprocessing import *

n_to_test = [250,500,750,1000,1250]

accuracy = []
f1_micro = []
f1_macro = []

for al_samplesize in n_to_test:
    # Parameters of the run
    kmeans = KMeans(
        init="random",
        n_clusters=al_samplesize,
        n_init=10,
        max_iter=300,
        random_state=42)

    kmeans.fit(embeddings_train)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings_train)
    
    embeddings_train_selected = embeddings_train[closest]
    labels_onehot_train_selected = labels_onehot_train[closest]

    weights = {0:1.0, 1:100.0}
    kmeans_svc = SVC(kernel="rbf", class_weight=weights, probability=True, random_state=42)

    multilabel_classifier_kmeans_svc = MultiOutputClassifier(kmeans_svc, n_jobs=-1)

    # We can only create labels if we have positive and negative examples, therefore we have to check which labels are usable
    valid_labels = [i for i in range(labels_onehot_train_selected.shape[1]) if len(np.unique(labels_onehot_train_selected[:,i])) > 1]

    # Select a subset with only the valid labels
    labels_onehot_train_selected_valid = labels_onehot_train_selected[:,valid_labels]

    # Fit the classifier with the valid labels
    multilabel_classifier_kmeans_svc.fit(
        embeddings_train_selected,
        labels_onehot_train_selected_valid)

    # Compute predictions on the test data for the valid labels 
    labels_onehot_test_predicted_valid = multilabel_classifier_kmeans_svc.predict(embeddings_test)

    # For Performance Evaluation we however need predidictions for all labels
    # Due to class imbalance, we simply predict 0 for all labels as a frist guess
    labels_onehot_test_predicted = np.zeros((labels_onehot_test_predicted_valid.shape[0], labels_onehot_train.shape[1]))

    # For our valid labels, we can update the dataframe with actual predictions
    labels_onehot_test_predicted[:,valid_labels] = labels_onehot_test_predicted_valid
    
    accuracy.append(accuracy_score(labels_onehot_test, labels_onehot_test_predicted))
    f1_micro.append(f1_score(labels_onehot_test, labels_onehot_test_predicted, average="micro"))
    f1_macro.append(f1_score(labels_onehot_test, labels_onehot_test_predicted, average="macro"))

    print(f"Results of the {al_samplesize}n run:")
    print(accuracy[-1])
    print(f1_micro[-1])
    print(f1_macro[-1])

results = {
    "n":n_to_test,
    "accuracy":accuracy,
    "f1_micro":f1_micro,
    "f1_macro":f1_macro}

# Serializing json
json_object = json.dumps(results, indent=4)

# Writing to sample.json
with open("results/kmsubset_svm_embeddings.json", "w") as outfile:
    outfile.write(json_object)