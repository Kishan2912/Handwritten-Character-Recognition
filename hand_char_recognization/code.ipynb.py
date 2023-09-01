# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
tf.random.set_seed(0)

trainX = []
trainY = []
testX = []
testY = []

for folder in os.listdir("./Handwriting_Data"):
    f = os.path.join("./Handwriting_Data", folder)

    tst = os.path.join(f, "dev")
    trn = os.path.join(f, "train")
    
    fnames = [tst, trn]
    for fname in fnames:
        for file in os.listdir(fname):
            file = os.path.join(fname, file)
            content = open(file, "r").read()
            content = content.split()[1:]
            content = np.array(content, dtype='float32')
            content = content.reshape(-1, 2)
            content = MinMaxScaler().fit_transform(content)
            if fname==tst:
                testX.append(content)
                testY.append(folder)
            else:
                trainX.append(content)
                trainY.append(folder)

# %%
trainY = np.asarray(trainY)
testY = np.asarray(testY)

# %%
from tensorflow.keras.preprocessing.sequence import pad_sequences

trainX_padded = pad_sequences(trainX, value=-1, dtype="float32")
testX_padded = pad_sequences(testX, value=-1, dtype="float32")

# %%
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

trainY = le.fit_transform(trainY)
testY = le.transform(testY)

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, Flatten

models = [
    Sequential([
        LSTM(128, input_shape=(None, 2)),
        Dense(16, activation='tanh'),
        Dense(5, activation='softmax')
    ]),
    Sequential([
        LSTM(128, return_sequences=True, input_shape=(None, 2)),
        LSTM(64),
        Dense(16, activation='tanh'),
        Dense(5, activation='softmax')
    ]),
    Sequential([
        LSTM(128, return_sequences=True, input_shape=(None, 2)),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(16, activation='tanh'),
        Dense(5, activation='softmax')
    ]),
    ####################
    Sequential([
        LSTM(128, dropout=0.3, input_shape=(None, 2)),
        Dense(16, activation='tanh'),
        Dense(5, activation='softmax')
    ]),
    Sequential([
        LSTM(128, return_sequences=True, dropout=0.3, input_shape=(None, 2)),
        LSTM(64, dropout=0.3),
        Dense(16, activation='tanh'),
        Dense(5, activation='softmax')
    ]),
    Sequential([
        LSTM(128, return_sequences=True, dropout=0.3, input_shape=(None, 2)),
        LSTM(64, dropout=0.3, return_sequences=True),
        LSTM(32, dropout=0.3),
        Dense(16, activation='tanh'),
        Dense(5, activation='softmax')
    ]),
    ####################
    Sequential([
        LSTM(128, dropout=0.5, input_shape=(None, 2)),
        Dense(16, activation='tanh'),
        Dense(5, activation='softmax')
    ]),
    Sequential([
        LSTM(128, return_sequences=True, dropout=0.5, input_shape=(None, 2)),
        LSTM(64, dropout=0.5),
        Dense(16, activation='tanh'),
        Dense(5, activation='softmax')
    ]),
    Sequential([
        LSTM(128, return_sequences=True, dropout=0.5, input_shape=(None, 2)),
        LSTM(64, dropout=0.5, return_sequences=True),
        LSTM(32, dropout=0.5),
        Dense(16, activation='tanh'),
        Dense(5, activation='softmax')
    ]),
    ########################################################################################
    Sequential([
        SimpleRNN(128, input_shape=(None, 2)),
        Dense(16, activation='tanh'),
        Dense(5, activation='softmax')
    ]),
    Sequential([
        SimpleRNN(128, return_sequences=True, input_shape=(None, 2)),
        SimpleRNN(64),
        Dense(16, activation='tanh'),
        Dense(5, activation='softmax')
    ]),
    Sequential([
        SimpleRNN(128, return_sequences=True, input_shape=(None, 2)),
        SimpleRNN(64, return_sequences=True),
        SimpleRNN(32),
        Dense(16, activation='tanh'),
        Dense(5, activation='softmax')
    ]),
    ####################
    Sequential([
        SimpleRNN(128, dropout=0.3, input_shape=(None, 2)),
        Dense(64, activation='tanh'),
        Dense(5, activation='softmax')
    ]),
    Sequential([
        SimpleRNN(128, return_sequences=True, dropout=0.3, input_shape=(None, 2)),
        SimpleRNN(64, dropout=0.3),
        Dense(16, activation='tanh'),
        Dense(5, activation='softmax')
    ]),
    Sequential([
        SimpleRNN(128, return_sequences=True, dropout=0.3, input_shape=(None, 2)),
        SimpleRNN(64, dropout=0.3, return_sequences=True),
        SimpleRNN(32, dropout=0.3),
        Dense(16, activation='tanh'),
        Dense(5, activation='softmax')
    ]),
    ####################
    Sequential([
        SimpleRNN(128, dropout=0.5, input_shape=(None, 2)),
        Dense(16, activation='tanh'),
        Dense(5, activation='softmax')
    ]),
    Sequential([
        SimpleRNN(128, return_sequences=True, dropout=0.5, input_shape=(None, 2)),
        SimpleRNN(64, dropout=0.5),
        Dense(16, activation='tanh'),
        Dense(5, activation='softmax')
    ]),
    Sequential([
        SimpleRNN(128, return_sequences=True, dropout=0.5, input_shape=(None, 2)),
        SimpleRNN(64, dropout=0.5, return_sequences=True),
        SimpleRNN(32, dropout=0.5),
        Dense(16, activation='tanh'),
        Dense(5, activation='softmax')
    ])
]

# %%
table_content = []

# %%
for index, model in enumerate(models):
    
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
        optimizer=tf.keras.optimizers.Adam(), 
        metrics=['accuracy'],
    )
    history = model.fit(trainX_padded, trainY, callbacks=[
             tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                min_delta=1e-4,
                mode="min",
                patience=1,
                start_from_epoch=2,
            ),
        ], epochs=1000, batch_size=32, verbose=0, shuffle=True)

    loss, accuracy = model.evaluate(testX_padded, testY, batch_size=1)
    epochs = len(history.history['loss'])
    
    table_content.append([index, history.history['loss'][-1], history.history['accuracy'][-1], accuracy, epochs])
    print(table_content[-1])
    
    print("MODEL :", index, "----------------------------")
    
    predictions = model.predict(testX_padded)
    predictions = tf.math.argmax(predictions, axis=1)
    cm = tf.math.confusion_matrix(testY, predictions, num_classes=5).numpy()
    cm = cm.astype(str)
    print("\n".join([f"&\\textbf{{{i+1}}} &" + " &".join(list(cm[i])) + "\\\\ \cline{3-7}" for i in range(5)]))
    print(cm)
    
    plt.plot(history.history['loss'], label="loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.savefig(f"/kaggle/working/{index}.png")
    plt.close()

# %%
s = ""

for c in table_content:
    s += f"{c[0]} & {c[1]:2f} & {c[2]:.2f}% & {c[3]:.2f}% & {c[4]} \\\\"

# %%
print(f"""
\begin{{table}}[]
\centering
\begin{{tabular}}{{lllll}}
\textbf{{Model No}} & \textbf{{Loss}} & \textbf{{Training Accuracy}} & \textbf{{Testing Accuracy}} & \textbf{{Epochs}} \\
{s}
\end{{tabular}}
\caption{{table caption}}
\label{{tab:my-table}}
\end{{table}}
""")

# %%
print(le.classes_)

# %%
trainX[0].shape

# %%
d = {0: [], 1: [], 2: [], 3: [], 4:[]}

index = 0
for i in trainY:
    if all([len(d[j])==5 for j in range(5)]):
        break
    t = trainX[index]
    d[i].append(t)
    index += 1
        
fig, ax = plt.subplots(5, 5, figsize=(7, 7))

cla = 0
index = 0
for ax_ in ax.flat:
    if index == 5:
        index = 0
        cla += 1
    x,y = d[cla][index].T
    ax_.scatter(x, y)
    ax_.axis('off')
    index += 1
plt.show()
plt.tight_layout()
plt.close()