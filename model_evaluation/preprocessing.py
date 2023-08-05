import numpy as np
import umap

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sentence_transformers import SentenceTransformer

# If True, Embeddings are created
CREATE_MODE = False

docs = []
root = "C:/Users/phili/Projekte/masterarbeit/al_zeroshot_perftrust_mb/"
with open(root + 'Test_Algos/reuters/reuters/cats.txt', encoding="ISO-8859-2") as f:
    docs = docs + f.readlines()
    # Remove newline
    for i, doc in enumerate(docs):
        docs[i] = doc.strip()

# 10788 Texts
# Is the document test or train?
test_train = []
# Index/Name of the document in the folder
doc_ind = []
# Labels
labels = []

# Split String from the cat file into these parts
for doc in docs:
    temp_test_train, temp_rest = doc.split("/")
    test_train.append(temp_test_train)
    temp_ind_labels_split = temp_rest.split()
    doc_ind.append(temp_ind_labels_split[0])
    labels.append(temp_ind_labels_split[1:])

# Get Index/Name for train and test data
train_ind = [doc_ind[i] for i, val in enumerate(test_train) if val == "training"]
test_ind = [doc_ind[i] for i, val in enumerate(test_train) if val == "test"]

# Loop over the documents 
def read_documents(indicies, location):
    temp = []
    for ind in indicies:
        with open(root + location + str(ind), encoding="ISO-8859-2") as d:
            temp.append(d.read())
    return temp

# Read in the documents
texts_train = read_documents(train_ind, "Test_Algos/reuters/reuters/training/")
texts_test = read_documents(test_ind, "Test_Algos/reuters/reuters/test/")

# Collect all labels
labels_flat = [val for lab in labels for val in lab]
labels_unique = np.unique(labels_flat)

# Convert to int for One hot Encoding
labels_to_int = {val: i for i, val in enumerate(labels_unique)}
int_to_labels = {i: val for i, val in enumerate(labels_unique)}

# Read in fullnames for Zero-Shot Classification
with open("cats_fullname.txt", "r") as f:
    cats_fullname = f.readlines()
    # Remove newline
    for i, val in enumerate(cats_fullname):
        cats_fullname[i] = val.strip()
# Some nasty text processing        
label_names_dic = {val.split("(")[1].strip(")"): val.split("(")[0] for val in cats_fullname}
labels_unique_fullname = [label_names_dic.get(val.upper()) for val in labels_unique]

# Create One Hot Encoding by having an all 0 vector 
labels_onehot = np.zeros((len(labels), len(labels_unique)))

# Adjust Position based on the label -> int mapping
for i, individ_labs in enumerate(labels):
    for lab in individ_labs:
        labels_onehot[i, labels_to_int.get(lab)] = 1

labels_onehot_train = np.zeros((len(texts_train), len(labels_unique)))
labels_onehot_test = np.zeros((len(texts_test), len(labels_unique)))

# Split labels into train and test 
i_train = 0
i_test = 0
for j, val in enumerate(test_train):
    if val == "training":
        labels_onehot_train[i_train,] = labels_onehot[j,]
        i_train += 1
    else:
        labels_onehot_test[i_test,] = labels_onehot[j,]
        i_test += 1

if CREATE_MODE:
    
    emb_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    embeddings_train = emb_model.encode(texts_train)
    embeddings_test = emb_model.encode(texts_test)
    embeddings_conc = np.concatenate((embeddings_train, embeddings_test))

    umap_obj = umap.UMAP(n_neighbors=5, min_dist=0.1, n_components=5)
    
    embeddings_conc_umap = umap_obj.fit_transform(embeddings_conc)
    embeddings_train_umap = embeddings_conc_umap[:len(texts_train),:]
    embeddings_test_umap = embeddings_conc_umap[len(texts_train):,:]
    
    # Save Embedings
    np.savetxt("reuters_train_embeddings.csv", embeddings_train, delimiter=",")
    np.savetxt("reuters_test_embeddings.csv", embeddings_test, delimiter=",")
    
    np.savetxt("reuters_train_embeddings_umap.csv", embeddings_train_umap, delimiter=",")
    np.savetxt("reuters_test_embeddings_umap.csv", embeddings_test_umap, delimiter=",")

else:
    embeddings_train = np.genfromtxt(root + "Test_Algos/reuters_train_embeddings.csv", delimiter=",")
    embeddings_test = np.genfromtxt(root + "Test_Algos/reuters_test_embeddings.csv", delimiter=",")

    embeddings_train_umap = np.genfromtxt(root + "Test_Algos/reuters_train_embeddings_umap.csv", delimiter=",")
    embeddings_test_umap = np.genfromtxt(root + "Test_Algos/reuters_test_embeddings_umap.csv", delimiter=",")
    
# Create tfidf matrix
vectorizer = CountVectorizer(max_features=5000)
count_vect = vectorizer.fit_transform(texts_train + texts_test)

count_vect_train = count_vect.toarray()[:len(texts_train),:]
count_vect_test = count_vect.toarray()[len(texts_train):,:]