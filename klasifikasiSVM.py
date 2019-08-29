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



categories = ['akibatGangguan', 'onset', 'keterangan', 'frekuensiSerangan', 'sifatSerangan', 'durasi', 'lokasi', 'perjalananPenyakit', 'riwayatPengobatan']
dataset = open("label.txt", 'r').read()
dataset = dataset.split('\n')
source = []
target = []
for token in dataset:
    token = token.split(',')
    source.append(token[0])
    target.append(token[1])

dataset2 = open("label2.txt", 'r').read()
dataset2 = dataset2.split('\n')
source2 = []
target2 = []
for token2 in dataset2:
    token2 = token2.split(',')
    source2.append(token2[0])
    target2.append(token2[1])

dataset3 = open("label3.txt", 'r').read()
dataset3 = dataset3.split('\n')
source3 = []
target3 = []
for token3 in dataset3:
    token3 = token3.split(',')
    source3.append(token3[0])
    target3.append(token3[1])
target3 = list(map(int, target3))
#feature extraction - creating a tf-idf matrix

def tfidf(data, ma = 0.6, mi = 0.0001):
    tfidf_vectorize = TfidfVectorizer(max_df = ma, min_df = mi)
    tfidf_data = tfidf_vectorize.fit_transform(data)
    return tfidf_data

#SVM classifier
# def test_SVM(x_train, x_test, y_train, y_test):
#     SVM = SVC(kernel = 'linear')
#     SVMClassifier = SVM.fit(x_train, y_train)
#     predictions = SVMClassifier.predict(x_test)
#     # csm = confusion_matrix(y_test, predictions)
#     a = accuracy_score(y_test, predictions)
#     p = precision_score(y_test, predictions, average = 'weighted')
#     r = recall_score(y_test, predictions, average = 'weighted')
#     return a, p, r

x_train, x_test, y_train, y_test = train_test_split(tfidf(source), target, test_size = 0.25, random_state = 0)

SVM = SVC(kernel = 'linear')
SVMClassifier = SVM.fit(x_train, y_train)
predictions = SVMClassifier.predict(x_test)

# Confusion Matrix
mat = confusion_matrix(y_test, predictions)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap="YlGnBu", xticklabels=categories, yticklabels=categories)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions, target_names=categories))

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

    print("TF IDF :")
    print(tfidf(dump))

    vectorizer = TfidfVectorizer()
    vectorizer.fit(dump)
    print("Vocab : ", vectorizer.vocabulary_)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(source)
    X_test_counts = count_vect.transform(dump)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    # SVM
    text_clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, n_iter=5)
    text_clf.fit(X_train_tfidf, target)

    predicted = text_clf.predict(X_test_tfidf)
    print("Prediksi : ", predicted)


    # colors = "brygcmkw"
    colors = {'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'}

    # X_train_tfidf = np.array(X_train_tfidf)
    # print(X_train_tfidf)
    # shuffle
    idx = np.arange(X_train_tfidf.shape[0])
    np.random.seed(13)
    np.random.shuffle(idx)
    X = X_train_tfidf[idx]
    print("XXXX", X)
    y = target3
    print("yyy", y)

    # # standardize
    mean = X.mean(axis=0)
    print("Mean", mean)
    # s = X.std
    # print("S", s)
    # std = X.std(axis=0)
    # X = (X - mean)/std

    h = .5  # step size in the mesh

    t_clf = SGDClassifier(alpha=0.001, max_iter=100, tol=1e-3).fit(X, y)
    # print("clf", t_clf)
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    print("HAHA", x_min, x_max)
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    print("HIHI", y_min, "HOHO", y_max)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    print("xx",xx, "yy", yy)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    # Z = text_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # print("Zzz", Z)

    # Put the result into a color plot
    # Z = Z.reshape(xx.shape)
    # cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    # plt.axis('tight')

    # Plot also the training points
    for i, color in zip(t_clf.classes_, colors):
        idx = np.where(y == i)
        plt.scatter(xx, yy, c=color, label=categories[i], cmap=plt.cm.Paired, edgecolor='white', s=20)
    plt.title("SVM")
    plt.axis('tight')

    # Plot the three one-against-all classifiers
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    coef = t_clf.coef_
    intercept = t_clf.intercept_

    def plot_hyperplane(c, color):
        def line(x0):
            return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]

        plt.plot([xmin, xmax], [line(xmin), line(xmax)],
                 ls="--", color=color)

    for i, color in zip(t_clf.classes_, colors):
        plot_hyperplane(i, color)
    plt.legend()
    plt.show()

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



