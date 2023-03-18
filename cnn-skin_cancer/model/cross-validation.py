
# # ----- ----- ----- ----- ----- -----
# # CROSS VALIDATION

# K.clear_session()
# del model
# del history


# # define 3-fold cross validation test harness
# kfold = KFold(n_splits=3, shuffle=True, random_state=11)

# cvscores = []
# for train, test in kfold.split(X_train, y_train):
#   # create model
#     model = build(lr=lr,
#                   init= init,
#                   activ= activ,
#                   optim=optim,
#                   input_shape= input_shape)

#     # Fit the model
#     model.fit(X_train[train], y_train[train], epochs=epochs, batch_size=batch_size, verbose=0)
#     # evaluate the model
#     scores = model.evaluate(X_train[test], y_train[test], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)
#     K.clear_session()
#     del model

# print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
