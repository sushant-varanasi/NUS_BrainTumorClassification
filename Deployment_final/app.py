import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from flask import Flask, redirect, url_for, render_template, request
# import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import nibabel as nib
import random
import numpy as np
ALLOWED_EXTENSIONS = {'nii'}
UPLOAD_FOLDER = './uploads'


app= Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    return layers.Activation("relu")(x)

    
class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)
    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

def pointnet():

    inputs = keras.Input(shape=(100000, 4))

    x = tnet(inputs, 4)
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    x = tnet(x, 32)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = layers.Dropout(0.3)(x)
    x = dense_bn(x, 128)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="pointnet")

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload', name=filename))
    return render_template("homepage.html")

@app.route("/upload/<name>", methods=['GET', 'POST'])
def upload(name):
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            #lisd=request.files.getlist("file")
            #filename=lisd[1]
            filename = file.filename
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('predict', name=filename))
    return render_template("homepage.html")


@app.route("/predict/<name>")
def predict(name):
    name='./uploads/'+name
    if name.endswith('_flair.nii'):
        flair=nib.load(name)
        seg=nib.load(name.replace('_flair.nii', '_seg.nii'))
    elif name.endswith('_seg.nii'):
        seg=nib.load(name)
        flair=nib.load(name.replace('_seg.nii','_flair.nii'))
    seg=seg.get_fdata()
    flair=flair.get_fdata()
    points=[]
    datmin=2000
    datmax=0
    for x in range(240):
        for y in range(240):
            for z in range(155):
                if seg[x][y][z]!=0 and flair[x][y][z]!=0:
                    points.append([x,y,z,flair[x][y][z]])
                if flair[x][y][z]<datmin:
                    datmin=flair[x][y][z]
                if flair[x][y][z]>datmax:
                    datmax=flair[x][y][z]
    for t in points:
        t[0]=t[0]/240
        t[1]=t[1]/240
        t[2]=t[2]/155
        t[3]=(t[3]-datmin)/(datmax-datmin) 
        
    if len(points)<100000:
        for j in range(len(points), 100000):
            points.append([0,0,0,0])                
    if len(points)>100000:
        points=random.sample(points, 100000)
    random.shuffle(points)
    points=np.array(points, dtype=float)
    points=points.reshape(1,100000,4)
    model=pointnet()
    model.load_weights('model1_weights.h5')
    prediction=model.predict(points)
    if(prediction>=0.5):
        grade='HGG'
    else:
        grade='LGG'
    np.array(points, dtype=float)
    return render_template("predict.html", name=name, grade= grade)

if __name__=="__main__":
    app.run()
