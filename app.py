from flask import Flask ,render_template ,request ,jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)
dic={0:'Mild Demented',1:'Moderate Demented',2:'Non Demented',4:' Very Mild Demented '}

model=load_model('hm21.h5')
model.make_predict_function()

def predict_label(img_path):
	# i = image.load_img(img_path, target_size=(150,150))
	# i = image.img_to_array(i)/255.0
	# i = i.reshape(1, 150,150,3)
	# p = model.predict_classes(i)
	# return dic[p[0]]
    img = image.load_img(img_path, target_size=(150, 150))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Make predictions using the loaded model
    predictions = model.predict(img)

    # Get the predicted class index
    predicted_class_idx = np.argmax(predictions[0])

    # Map the predicted class index to the corresponding label
    return dic[predicted_class_idx]


@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)