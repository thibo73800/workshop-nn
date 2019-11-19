from tkinter import *
import PIL
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# WARNING : The order of the following list might change on your computer
# You should print the labels in your quick_draw file to get your list.
labels  = ['book', 'eye', 'car', 'brain', 'apple', 'face', 'airplane', 'The Eiffel Tower', 'dog', 'chair']

def process_image(img):
    img = img.resize((28, 28))
    img = img.convert('LA')
    img = np.array(img)
    img = img[:,:,0]
    img = img.reshape((1, 28, 28, 1))
    img = img / 255
    return img

def predict():
    img = process_image(pic)
    class_name = ""

    # TODO : Load the model and run a prediction
    # Use the np.argmax method to get the class with the highest probability
    # and print the prediction result of your model
    print("class_name", class_name)

def activate_paint(e):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint)
    lastx, lasty = e.x, e.y

def paint(e):
    global lastx, lasty
    x, y = e.x, e.y
    cv.create_line((lastx, lasty, x, y), width=20)
    #  --- PIL
    draw.line((lastx, lasty, x, y), fill='black', width=20)
    lastx, lasty = x, y

root = Tk()
lastx, lasty = None, None
image_number = 0
cv = Canvas(root, width=280, height=280, bg='white')
# --- PIL
pic = PIL.Image.new('RGB', (280, 280), 'white')
draw = ImageDraw.Draw(pic)
cv.bind('<1>', activate_paint)
cv.pack(expand=YES, fill=BOTH)
btn_predict = Button(text="predict", command=predict)
btn_predict.pack()
root.mainloop()
