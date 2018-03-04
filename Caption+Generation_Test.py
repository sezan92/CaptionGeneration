
# coding: utf-8

# ## Caption Generation of Images.
# #### MD Muhaimin Rahman
# ##### contact : sezan92[at]gmail[dot]com
# In this notebook, I have tried to work on Caption generation of Images of Flickr_8k dataset. The notebook took extensive help from Jason Brownlee's Blog [article](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/) on the same dataset. But I thought some codeblocks were unnecessarily complex . So I changed them for this project

# #### Importing Libraries

# In[1]:


import keras
import pickle
from keras.applications.vgg16 import VGG16
from keras.applications import Xception,InceptionV3,InceptionResNetV2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# #### Extracting features from VGG16, XCeption, Inception version 3, 2 , saving them in pickle files respectively 

# In[2]:


def prev_model():
    model= VGG16()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    #model.summary()
    return model


# In[3]:





vocab = pickle.load(open('vocabulary.pkl'))


# In[22]:


vocab


# In[23]:


vocab_size=len(vocab)+1
print('Vocabulary Size %d'%vocab_size)


# In[24]:


i2w = dict((i,c)for i,c in enumerate(vocab))
w2i = dict((c,i)for i,c in enumerate(vocab))


# In[25]:


w2i['raining']


# Maximum length

# In[26]:


max_length =35
print("Maximum Length %d"%(max_length))


# In[38]:


from keras.models import model_from_json


# In[39]:


json_file = open('output/Caption_model_VGG16.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("output/Caption_model_VGG16.h5")
print("Loaded model from disk")


# In[40]:

model =prev_model()
def get_img_feature(filename,model=model):
    
    img = load_img(filename,target_size=(224,224))
    img = img_to_array(img)
    img = preprocess_input(img)
    img = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
    feature = model.predict(img)
    return feature


# In[41]:
test_dir ='Propic'
image_names =os.listdir(test_dir)

for image_name in image_names:
    
    image_name_full = test_dir+'/'+image_name

    features = get_img_feature(image_name_full)
    in_text ='start'
    in_text_encode = [w2i[in_text]]

    seq = pad_sequences([in_text_encode],maxlen=max_length)
    output =[]
    print(image_name)
    for i in range(max_length): 
        yoh = loaded_model.predict(x=[features,seq])
        word_indice = np.argmax(yoh)
        if i2w[word_indice]=='end.':
            continue
        else:
        
            output.append(i2w[word_indice])
            seq = seq.tolist()[0]
            seq.remove(seq[0])
            seq.append(word_indice)
            seq =np.array([seq])
        
    out_text =' '.join(output)
    im = plt.imread(image_name_full)
    plt.figure()
    plt.imshow(im)
    plt.title(out_text)
    plt.savefig('Caption%d.jpg'%(image_names.index(image_name)))
