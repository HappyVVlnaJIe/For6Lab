from pathlib import Path

import numpy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Bidirectional,TimeDistributed,Flatten
from keras.layers import LSTM, SpatialDropout1D



#dataset_path=Path.cwd()
#file_path = dataset_path / trainForOrfWithRand.txt
xTrainDecode=[]
xTestDecode=[]
y_train=[]
y_test=[]
vowels_mas=["а","е","ё","и","о","ю","я","у","э","ы"]
f=open(r"trainForOrfWithRand.txt","r",encoding='utf-8')
#f=file_path.open(encoding='utf-8')
while True:
    line=f.readline()
    x=line
    if len(line)==0:
        break
    str=line.split()
    xTrainDecode.append(str[0])
    #y_train.append(str[1])
    mas=[]
    #k=int(str[1])
    for i in range(len(str[0])):
       if (i==int(str[1])):
           mas.append(1)
       else: mas.append(0)
    y_train.append(mas)

    line = f.readline()
    y = line
    if len(line) == 0:
        break
    str=line.split()
    xTestDecode.append(str[0])
    mas = []
    #k = int(str[1])
    for i in range(len(str[0])):
       if (i==int(str[1])):
           mas.append(1)
       else: mas.append(0)
    y_test.append(mas)

f.close()

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(xTestDecode+xTrainDecode)
x_train = tokenizer.texts_to_sequences(xTrainDecode)
x_test = tokenizer.texts_to_sequences(xTestDecode)

x_train = sequence.pad_sequences(x_train, maxlen=32)
x_test = sequence.pad_sequences(x_test, maxlen=32)
y_train = sequence.pad_sequences(y_train, maxlen=32)
y_test = sequence.pad_sequences(y_test, maxlen=32)

model=Sequential()
model.add(Embedding(36,8,input_length=32))
model.add(Bidirectional(LSTM(50,return_sequences=True)))
model.add(TimeDistributed(Dense(1)))
model.add(Flatten())
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=1, shuffle=False)

scores = model.evaluate(x_test, y_test,batch_size=32)
print("Точность на тестовых данных: %.2f%%" % (scores[1] * 100))

#result=model.predict(x_test,batch_size=32)
#k=0
#for j in range(result.length):
 #   max=0
  #  indexMax=0
   # indexlastVowel=0
    #indexVowel=0
    #sumBefore=0
    #sumAfter=0
    #for i in range(result[j].length):
     #   if (xTestDecode[j][i] in vowels_mas):
      #      indexVowel=i
       # if (result[j][i]>max):
        #    sumBefore+=sumAfter
         #   max=result[j][i]
          #  indexMax=i
           # indexlastVowel=indexVowel
            #sumAfter=max
        #else:
         #   sumAfter+=result[j][i]
    #sumAfter-=max
    #if (xTestDecode[j][indexMax] not in vowels_mas):
     #   if (sumBefore>sumAfter and indexlastVowel!=0):
      #      for i in range(result[j].length):
       #         if (i==indexlastVowel):
        #            result[j][i]=1
         #       else:
          #          result[j][i]=0
        #else:
         #   newIndexMax=indexMax
          #  while (xTestDecode[j][newIndexMax] not in vowels_mas):
           #     newIndexMax+=1
            #for i in range(result[j].length):
             #   if (i==newIndexMax):
              #      result[j][i]=1
               # else:
                #    result[j][i]=0
    #if (result[j]==y_test[j]):
     #   k+=1
#print("Точность на тестовых данных: %.2f%%" % (k * 100))