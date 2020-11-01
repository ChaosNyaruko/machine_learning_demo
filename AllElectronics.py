from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
#from sklearn.externals import StringIO

allElectronicsData = open(r'/home/nyaruko/ml/test1.csv')
reader = csv.reader(allElectronicsData)
# headers = reader.next()
i = 0
labelList=[]
featureList=[]
headers=[]
print("read again")
for row in reader:
    if i == 0:
        i=i+1
        headers=row
        continue
    print(row)
    labelList.append(row[-1])
    rowDict={}
    for j in range (1, len(row)-1):
        rowDict[headers[j]] = row[j]
    i=i+1
    featureList.append(rowDict)

print(featureList)

vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
print(dummyX)
print(vec.get_feature_names())

print(labelList)
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print(dummyY)

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print("clf", str(clf))


oneRowX=dummyX[0, :]
print(oneRowX)
oneRow = oneRowX.reshape(1,-1)
print(oneRow)
predictedY=clf.predict(oneRow)
print("predictedY", predictedY)
