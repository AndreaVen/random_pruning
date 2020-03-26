from keras.applications.vgg16 import VGG16
model = VGG16()
print(model.summary())
img=np.random.randn(1,224,224,3)
y=model.predict(img)
print(np.argmax(y))
  tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
   
   # mean = np.mean(x_train_reduced,axis=(0,1,2,3))
   # std = np.std(x_train_reduced, axis=(0, 1, 2, 3))
   # x_test_reduced = (x_test_reduced-mean)/(std+1e-7)
   # idx=1
   # true_label=y_test_reduced[idx]
   # predicted=model.predict(x_train_reduced)
   # idx=5
   # data=x_test_reduced[idx]
   # predicted=model.predict(x_train_reduced, y_train_reduced, x_test_reduced, y_test_reduced, x_val_reduced, y_val_reduced, 3)
   # directly_predicted=predicted[idx]
   # print(directly_predicted)
   
   
    
    # Fetch the Keras session and save the model
    # The signature definition is defined by the input and output tensors
    # And stored with the default serving key
   export_path = 'R:\\mio_modello'
   keras.backend.get_session().run(tf.global_variables_initializer())
   sess=K.get_session()
   tf.saved_model.simple_save(sess,export_path,inputs={'input_image': model.input},outputs={t.name: t for t in model.outputs})
