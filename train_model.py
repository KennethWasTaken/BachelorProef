from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

path_embeddings = r"./result/embeddings.pickle"
path_recognizer = r"./output/recognizer.pickle"
path_label_encoder= r"./output/le.pickle"

# load the face embeddings
print("get face embeddings")
data = pickle.loads(open(path_embeddings, "rb").read())

# encode the labels
print("label laddle label")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition

print("[train model for 128 d features")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# write face recognition model to disk
print("face the facemodel model facer")
f = open(path_recognizer , "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open(path_label_encoder, "wb")
f.write(pickle.dumps(le))
f.close()