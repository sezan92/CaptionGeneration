
import keras
import pickle
from keras.applications.vgg16 import VGG16
#from keras.applications import Xception,InceptionV3,InceptionResNetV2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
#from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
import numpy as np
import pickle
import os

def prev_model():
    model= VGG16()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    model.summary()
    return model

directory = 'Flicker8k_Dataset'

def extract_features(directory,model_name):
    if os.path.exists('dataset/features_%s.pkl'%(model_name)):
        print("Features file already exists")
        features = pickle.load(open('dataset/features_%s.pkl'%(model_name)))
        return features
    else:
        model = prev_model()
        image_names = os.listdir(directory) 
        features = dict()
        for image_name in image_names:
            image =load_img(directory+'/'+image_name,target_size=(224,224))
            image = img_to_array(image)
            image = preprocess_input(image)
            image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
            feature = model.predict(image)
            features[image_name] = feature

            print('%s done!'%(image_name))
        pickle.dump(features,open('dataset/features_%s.pkl'%(model_name),'w')) 
        return features

features = extract_features(directory,'VGG16')

features.values()[0].shape

def extract_captions(filename):
    if os.path.exists('dataset/data_with_captions.pkl'):
        print('Data with Captions file already exists')
        dataset = pickle.load(open('dataset/data_with_captions.pkl'))
        return dataset
    else:
        text = open(filename).read()
        text = text.split(('\n'))
        captions=[]
        image_names=[]
        dataset=dict()
        for line in text:
            if len(line)<1:
                break
            #print(line.split('\t')[0].split('#')[0])
            caption = line.split('\t')[1]
            image_name = line.split('\t')[0].split('#')[0]
            if image_name not in dataset:
                dataset[image_name] =[]
            else:
                dataset[image_name].append(caption.lower())
                #dataset[image_name] =' '.join(dataset[image_name])
            print(image_name+" done!")
        pickle.dump(dataset,open('dataset/data_with_captions.pkl','w'))

        return dataset

filename= 'dataset/Flickr_Text/Flickr8k.token.txt'

data_with_captions= extract_captions(filename)

train_filename = 'dataset/Flickr_Text/Flickr_8k.trainImages.txt' 
dev_filename='dataset/Flickr_Text/Flickr_8k.devImages.txt'
test_filename = 'dataset/Flickr_Text/Flickr_8k.testImages.txt'


def load_dataset(filename):
    image_names = open(filename).read().split('\n')
    for image_name in image_names:
        if len(image_name)<1:
            image_names.remove(image_name)
    return image_names

train = load_dataset(train_filename)
dev= load_dataset(dev_filename)
test=load_dataset(test_filename)

def get_features(dataset):
    features_dict= dict()
    features = pickle.load(open('dataset/features_VGG16.pkl'))
    for image_name in dataset:
        features_dict[image_name]= features[image_name]
    return features_dict

train_features_set = get_features(train)
test_features_set = get_features(test)
dev_features_set = get_features(dev)

for desc_list in data_with_captions.values():
    for d in desc_list:
        desc_list[desc_list.index(d)] = 'start '+d+' end.'

def load_desc(data):
    desc =dict()
    for image_name in data:
        desc[image_name]=data_with_captions[image_name]
    return desc

train_desc = load_desc(train)
dev_desc = load_desc(dev)
test_desc = load_desc(test)

train_desc

texts = data_with_captions.values()
texts = ' '.join([' '.join(text) for text in texts])

texts_list = texts.split()

vocab = sorted(set(texts_list))

vocab

vocab_size=len(vocab)+1
print('Vocabulary Size %d'%vocab_size)

i2w = dict((i,c)for i,c in enumerate(vocab))
w2i = dict((c,i)for i,c in enumerate(vocab))

w2i['raining']

max_length =max(max([[len(d.split()) for d in ls] for ls in train_desc.values()]))

print("Maximum Length %d"%(max_length))

def encode_desc(description):
#if True:
    #description = train_desc
    encoded_list = []
    encoded_list_extend=[]
    out_list=[]
    encoded =dict()
    for key in description.keys():
        caps_encoded =[[w2i[word] for word in cap.split()] for cap in description[key]]
        encoded[key] = caps_encoded
        for cap in description[key]:
            encoded_list.append([w2i[word] for word in cap.split()])
    for ls in encoded_list:
        j=1
        for i in range(1,len(ls)):
            word_encode= pad_sequences([ls[:i]],maxlen=max_length,padding='pre')
            out = ls[i]
            encoded_list_extend.append(word_encode.tolist())
            out_list.append(out)
            print('.'*j+'\r'),
            j=j+1
    return encoded,encoded_list_extend,out_list

train_desc_encoded,train_desc_encoded_list,train_out = encode_desc(train_desc)
dev_desc_encoded,dev_desc_encoded_list,dev_out = encode_desc(dev_desc)
test_desc_encoded,test_desc_encoded_list,test_out = encode_desc(test_desc)

train_desc_encoded

train_desc_encoded_list

train_desc_encoded_np = np.array(train_desc_encoded_list)
dev_desc_encoded_np = np.array(dev_desc_encoded_list)
test_desc_encoded_np = np.array(test_desc_encoded_list)
print("Training array shape "+str(train_desc_encoded_np.shape))
print("Dev array shape "+str(dev_desc_encoded_np.shape))
print("Test array shape "+str(test_desc_encoded_np.shape))

train_desc_encoded_np = np.reshape(train_desc_encoded_np,(-1,train_desc_encoded_np.shape[2]))
dev_desc_encoded_np = np.reshape(dev_desc_encoded_np,(-1,dev_desc_encoded_np.shape[2]))
test_desc_encoded_np = np.reshape(test_desc_encoded_np,(-1,test_desc_encoded_np.shape[2]))
print("Training array shape "+str(train_desc_encoded_np.shape))
print("Dev array shape "+str(dev_desc_encoded_np.shape))
print("Test array shape "+str(test_desc_encoded_np.shape))

train_out_np = np.array(train_out)
dev_out_np = np.array(dev_out)
test_out_np = np.array(test_out)
print("Training Output array shape "+str(train_out_np.shape))
print("Dev Output array shape "+str(dev_out_np.shape))
print("Test Output array shape "+str(test_out_np.shape))

def prepare_features(features_set,description_encoded):
    x1=[]

    #features_set = train_features_set
    #description_encoded = train_desc_encoded
    for key,values in features_set.items():
        photo_descs = description_encoded[key]

        j=0
        for desc in photo_descs:
            j=j+1
            for i in range(1,len(desc)):

                #in_seq = pad_sequences([desc[:i]],maxlen=max_length)[0]
                #out_seq = np_utils.to_categorical(desc[i],num_classes=len(vocab)+1)[0]
                x1.append(features_set[key][0])
                #x2.append(in_seq)
                #y.append(out_seq)

    return x1

train_x1 = prepare_features(train_features_set,train_desc_encoded)
dev_x1 = prepare_features(dev_features_set,dev_desc_encoded)
test_x1 = prepare_features(test_features_set,test_desc_encoded)


np.array(train_x1[1:10]).shape
print("Defining Model!")
if True:
    
    in1  = Input(shape=(4096,))
    f1 = Dropout(0.5)(in1)
    f2 = Dense(256,activation='relu')(f1)
    in2 = Input(shape=(max_length,))
    em2 =Embedding(vocab_size,256,mask_zero=True)(in2)
    d2 = Dropout(0.5)(em2)
    lstm = LSTM(256)(d2)
    dec1 = add([f2,lstm])
    dec2 = Dense(256,activation='relu')(dec1)
    output = Dense(vocab_size,activation='softmax')(dec2)
    full_model = Model(inputs=[in1,in2],outputs=output)
    full_model.summary()
    full_model.compile(loss='sparse_categorical_crossentropy',optimizer='adam')
print("Defined!")
plot_model(full_model, to_file='model_new.png', show_shapes=True)

len(train_x1)
batch_size=1024
#dev_x1 = dev_x1[:-batch_size]
epochs = 20
filepath = 'output/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
print("Starting to train....")
#checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# fit model
for epoch in range(epochs):
    for i in range(0,len(train_x1),batch_size):
        X1train = np.array(train_x1[i:i+batch_size])
        X2train = train_desc_encoded_np[i:i+batch_size]
        ytrain = train_out_np[i:i+batch_size]
        X1test = np.array(dev_x1[i:i+batch_size])
        X2test = dev_desc_encoded_np[i:i+batch_size]
        ytest = dev_out_np
        full_model.fit([X1train, X2train], ytrain, verbose=1,batch_size=batch_size,validation_data=([X1test, X2test], ytest))
    train_loss =full_model.evaluate(x=[X1train, X2train], y=ytrain,verbose=0)
    Val_loss = full_model.evaluate(x=[X1test, X2test], y=ytest,verbose=0)
    print("Epoch %d , Train Loss %f and Val Loss %f"%(epoch,train_loss,Val_loss))

#full_model.evaluate(x=[X1train,X2train],y=ytrain)
full_model.evaluate(x=[X1test,X2test],y=ytest)

model_json = full_model.to_json()
with open("output/Caption_model_VGG16.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
full_model.save_weights("output/Caption_model_VGG16.h5")
print("Saved model to disk")
