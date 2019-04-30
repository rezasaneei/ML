import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import distance


def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile,'r',encoding='utf8')
    print ("File loaded")    
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model

model=loadGloveModel('glove.6B/glove.6B.50d.txt')
print(model['hello'])

vec1=model['dog']
vec2=model['cat']
vec3=model['book']
dist = np.linalg.norm(vec1-vec2)
dist2=np.linalg.norm(vec1-vec3)



print(dist)
print(dist2)
sentence_conf=[[],[],[]]
sentence2=['i','do','like','computer','too','much']
sentence1=['i','like','computer']

for i in sentence1:
    empty_col=[]
#    sentence_conf.append(empty_col)
    vec1=model[i]
    for j in sentence2:
        vec2=model[j]
#        dist=int((np.linalg.norm(vec2-vec1)+1)*100)
#        dist=np.linalg.norm(vec2-vec1)
        dist = distance.euclidean(vec2,vec1)

        sentence_conf[sentence1.index(i)].append(dist)
        
print(sentence_conf)
        

        



#conf_arr = [[33,2,0,0,0,0,0,0,0,1,3], 
#            [3,31,0,0,0,0,0,0,0,0,0], 
#            [0,4,41,0,0,0,0,0,0,0,1], 
#            [0,1,0,30,0,6,0,0,0,0,1], 
#            [0,0,0,0,38,10,0,0,0,0,0], 
#            [0,0,0,3,1,39,0,0,0,0,4], 
#            [0,2,2,0,4,1,31,0,0,0,2],
#            [0,1,0,0,0,0,0,36,0,2,0], 
#            [0,0,0,0,0,0,1,5,37,5,1], 
#            [3,0,0,0,0,0,0,0,0,39,0], 
#            [0,0,0,0,0,0,0,0,0,0,38]]
#
conf_arr=sentence_conf
a = 0
for i in conf_arr:
    for j in i:
        a = a+j
        
norm_conf = []
for i in conf_arr:
#    a = 0
    tmp_arr = []
#    a = sum(i, 0)
    for j in i:
        tmp_arr.append(float(j)/float(a))
    norm_conf.append(tmp_arr)
#fig = plt.figure()
fig=plt.figure(num=None, figsize=(24, 9), dpi=80, facecolor='w', edgecolor='k')

plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(norm_conf), cmap='gray_r', 
                interpolation='nearest')
plt.savefig('confusion_matrix.png', format='png')
width, height = conf_arr.shape

for x in xrange(width):
    for y in xrange(height):
        ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center')

cb = fig.colorbar(res)
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
plt.xticks(range(width), alphabet[:width])
plt.yticks(range(height), alphabet[:height])
plt.savefig('confusion_matrix.png', format='png')