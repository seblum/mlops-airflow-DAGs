# ----- ----- ----- ----- ----- -----
## RESNET 50

input_shape = (224, 224, 3)
lr = 1e-5
epochs = 50
batch_size = 64

model = ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=input_shape, pooling="avg", classes=2)

model.compile(optimizer=Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=epochs,
    batch_size=batch_size,
    verbose=2,
    callbacks=[learning_rate_reduction],
)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()
# summarize history for loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

# Train ResNet50 on all the data
model.fit(X_train, y_train, epochs=epochs, batch_size=epochs, verbose=0, callbacks=[learning_rate_reduction])

# Testing model on test data to evaluate
y_pred = model.predict(X_test)
print(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))

# save model
# serialize model to JSON
resnet50_json = model.to_json()

with open("resnet50.json", "w") as json_file:
    json_file.write(resnet50_json)

# serialize weights to HDF5
model.save_weights("resnet50.h5")
print("Saved model to disk")
# 0.8287878787878787
