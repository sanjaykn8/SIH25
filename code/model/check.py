# run in Python where tensorflow is available
import tensorflow as tf
m = tf.lite.Interpreter(model_path="C:/Users/nisan/Desktop/code/sih/code/model/microplastic_image_classifier_antioverfit.tflite")
m.allocate_tensors()
d = m.get_input_details()[0]
print("dtype:", d['dtype'])
print("shape:", d['shape'])
print("quantization:", d['quantization'])  # (scale, zero_point)
