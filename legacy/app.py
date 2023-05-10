def pre_process_img_v3(source_filepath):
    left, upper, right, lower = 170, 200, 350, 380
    with Image.open(source_filepath) as img:
        cropped_img = img.crop((left, upper, right, lower))

    grey_img = cropped_img.convert('L')
    
    grey_img = grey_img.point(lambda p: 255 if p > 165 else p)
    grey_img = grey_img.point(lambda p: 110 if p < 130 else p)

    np_img = np.asarray(grey_img)
    return np_img

def predict_with_tflite(image_path, model_path):
    image = pre_process_img_v3(image_path,)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    image = image.astype(np.float32)

    # Load the model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get the input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Make the prediction
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    # Return the prediction
    prediction_accuracy = max(prediction[0])
    class_name = prediction_classes[np.argmax(prediction)]

    return [class_name, prediction_accuracy]