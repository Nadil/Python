from boto import sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.svm import SVC
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


#nama kelas
categories = ['akibatGangguan', 'onset', 'keterangan', 'frekuensiSerangan', 'sifatSerangan', 'durasi', 'lokasi', 'perjalananPenyakit', 'riwayatPengobatan']

#training
dataset = open("label.txt", 'r').read()
dataset = dataset.split('\n')
source = []
target = []
for token in dataset:
    token = token.split(',')
    source.append(token[0])
    target.append(token[1])

    #testing
dataset2 = open("label2.txt", 'r').read()
dataset2 = dataset2.split('\n')
source2 = []
target2 = []
for token2 in dataset2:
    token2 = token2.split(',')
    source2.append(token2[0])
    target2.append(token2[1])

def predictSVM(datates):
    onset = []
    keterangan = []
    frekuensiSerangan = []
    sifatSerangan = []
    durasi = []
    lokasi = []
    perjalananPenyakit = []
    riwayatPengobatan = []
    akibatGangguan = []
    dump = []
    for i in datates:
        dump.append(i[0])

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(source)
    print("X_train tfidf", X_train_counts)
    X_test_counts = count_vect.transform(dump)
    print("X test count", X_test_counts)
    
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print("X_train_tfidf", X_train_tfidf)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    print("X_tes_tfidf", X_test_tfidf)

    #klasifikasi SVM
    text_clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, n_iter=5)
    text_clf.fit(X_train_tfidf, target)
    predicted = text_clf.predict(X_test_tfidf)
    print("Prediksi : ", predicted)

    
    t = 0
    hitung = 0
    while t < len(predicted):
        if predicted[t] != target2[t]:
            hitung += 1
            print(str(predicted[t]) + "-----" + str(target2[t]))
        else:
            print(str(predicted[t]) + "-----" + str(target2[t]))
        t += 1
    print("Tidak sesuai : " +str(hitung))
    presentase = round(((len(predicted)-hitung)/len(target2)),3)
    print(str("%f" % (presentase*100) + "%"))
    a = accuracy_score(target2, predicted)
    print("Akurasi : ", str("%f" % (a * 100) + "%"))

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    mat = confusion_matrix(target2, predicted)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap="YlGnBu", xticklabels=categories, yticklabels=categories)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()

    from sklearn.metrics import classification_report
    print(classification_report(target2, predicted))
    
   
    for doc, pred in zip(datates, predicted):
        print(doc, pred)
        if pred == "onset":
            onset.append(doc)
        elif pred == "keterangan":
            keterangan.append(doc)
        elif pred == "frekuensiSerangan":
            frekuensiSerangan.append(doc)
        elif pred == "sifatSerangan":
            sifatSerangan.append(doc)
        elif pred == "durasi":
            durasi.append(doc)
        elif pred == "lokasi":
            lokasi.append(doc)
        elif pred == "perjalananPenyakit":
            perjalananPenyakit.append(doc)
        elif pred == "riwayatPengobatan":
            riwayatPengobatan.append(doc)
        elif pred == "akibatGangguan":
            akibatGangguan.append(doc)
    return onset, keterangan, frekuensiSerangan, sifatSerangan, durasi, lokasi, perjalananPenyakit, riwayatPengobatan, akibatGangguan



