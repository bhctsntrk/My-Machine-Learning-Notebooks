# -*- coding: utf-8 -*-
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

#Data setinin öznitelikleri çıkarılmış durumda
#29 adet öznitelik var "Class" ise çıktı kısmı

#csv dosyasından verileri alıyoruz pandas kütüphanesi csv okumada işimize yarıyor
veri = pd.read_csv('~/Desktop/creditcard.csv', header=0)
#veriyi ayrı bir yere kopyalıyoruz
veri_kopya = veri.copy()
#Daha sonra girdi ve çıktıları ayıtıp düzenliyoruz 
ciktilar = veri_kopya['Class'].values.tolist()
del veri_kopya['Class']
girdiler = veri_kopya.values.tolist()
ciktilar = to_categorical(ciktilar, 2)

#SkLearn kütüphanesi yardımı ile tüm data'nın 0.2'sini Test verisi
#olarak ayırıyoruz.
X_train, X_test, y_train, y_test = train_test_split(girdiler, ciktilar, test_size=0.2)

#Sequential bir model oluşturuyoruz.Üç katmanlı bir DNN kuruyoruz
model = Sequential()
#İlk katman 50 nöron içerecek ve aktivasyon fonksiyonu olarak relu kullanacağız
model.add(Dense(50, input_dim=30, activation='relu'))
#Hemen sonra dropout koyuyoruz böylece overfit'in önüne geçmeye çalışıyoruz
model.add(Dropout(0.5))
#Bir katman daha
model.add(Dense(50, activation='relu'))

model.add(Dropout(0.5))
#Son olarak çıktı katmanı'nı ayarlıyoruz.İki tür çıktı var ve
#ve aktivasyon fonksiyonu olarak softmax kullanıyoruz
model.add(Dense(2, activation='softmax'))
#Hata fonkiyonu olarak cat_crossentropy kullanıyoruz MSE ya da başka
#bir şeyde kullanabiliriz.Optimizer fonksiyonu SGD olacak
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.sgd(),
              metrics=['accuracy'])
#Eğitimi başlatıyoruz.Epoch'u 10'a ayarlıyoruz sistem 10 kez kendini tekrar edecek
#ve kendini eğitecek
model.fit(X_train, y_train,
          batch_size=100,
          epochs=10,
          validation_data=(X_test, y_test))

#Başarı durumunu hesaplayıp ekrana bastırıyoruz          
basari = model.evaluate(X_test, y_test)
print('Hata Toplami:', basari[0])
print('Basari:', basari[1])
