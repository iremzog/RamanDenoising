# Raman Signal Denoising using Fully Convolutional Encoder Decoder Network


Models directory contains three models. Two of them for the proof of concept work (model_mse_loss.h5 and model_custom_loss.h5) and the other one is for application purposes (model_application.h5). 

In order to use any model, ```alpha``` and ```thr_ratio``` parameters must be specified in the custom loss function. Alpha is the binary parameter specifying whether local MSE loss is included in the loss funtion. ```alpha=0``` is the regular MSE loss. For the model_mse_loss.h5 and model_application.h5, select ```alpha=0```. For the model_custom_loss.h5 select ```alpha=0``` and ```thr_ratio=0.25```.

Example usage:
    
    model = tf.keras.models.load_model(model_path, custom_objects={'peak_mse_combined_loss':custom_loss(alpha=ALPHA, thr_ratio=THRESHOLD)})
