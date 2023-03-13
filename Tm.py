from keras.models import load_model  
import cv2  
import numpy as np
from tkinter import font
import tkinter
from tkinter import *
from tkinter.ttk import *
import PIL.Image, PIL.ImageTk
from tkinter import filedialog


np.set_printoptions(suppress=True)

model = load_model("C:\HinhTanHiep\keras_model.h5", compile=False)

class_names = open("C:\HinhTanHiep\labels.txt", "r").readlines()



def giaodien():
                              
    root=Tk()
    root.title("Nhận Diện Xe Máy")
    root.geometry("700x700")
    root.config(bg="#BCE6FF")
    button_frame= Frame(root).pack(side=BOTTOM)
    
     
    sign_image = tkinter.Label(root)
      
    def nhandien():
        camera = cv2.VideoCapture(0)       
        while True:
            
            ret, image = camera.read()
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
                       
            cv2.imshow("Webcam Image", image)
            
            image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

            image = (image / 127.5) - 1

            prediction = model.predict(image)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

                      
            xe="".join(["Xe:",class_name[2:]])
            accuracy="".join(["Accuracy:",str(np.round(confidence_score * 100))[:-2],"%" ])                           
            print("Class:", class_name[2:], end="")
            print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")            
            keyboard_input = cv2.waitKey(1)            
            if keyboard_input == 27:                                              
                break           
        update_frame(xe,accuracy)
        cv2.destroyAllWindows()
    
    def anh(file_path):     
        image = PIL.Image.open(fp=file_path,mode="r") 
        image = image.resize((224, 224)) 
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image / 127.5) - 1
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        xe="".join(["Xe:",class_name[2:]])
        accuracy="".join(["Accuracy:",str(np.round(confidence_score * 100))[:-2],"%" ])                           
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
        update_frame(xe,accuracy)
                       
    def update_frame(xe,accuracy):
        vatthe.config(text=xe)
        acc.config(text=accuracy)
             
    def upload_image():
        try:
            file_path = filedialog.askopenfilename()
            uploaded = PIL.Image.open(fp=file_path,mode="r")
            
            uploaded.thumbnail(((root.winfo_width() / 2.25), (root.winfo_height() / 2.25)))
            im = PIL.ImageTk.PhotoImage(uploaded)

            sign_image.config(image=im)
            sign_image.image_names = im
            anh(file_path)
        except:
            pass


    upload = tkinter.Button(button_frame,text="UpLoad",font=(("Arial"),10,'bold'),bg="#303030",fg="#FFFF00",command=upload_image )
    upload.pack(padx=10)
    upload.pack(pady=0) 
    upload.pack(side=BOTTOM, pady=50)
    
    sign_image.pack(side=BOTTOM, expand=True)
    
    
    lable = tkinter.Label(root,text="Ấn Nhận Diện hoặc UpLoad để bắt đầu",fg="black",bd=0,bg="#E0ECDE" )
    lable.config(font=("",10))
    lable.pack(pady=0)
    
    lable1 = tkinter.Label(root,text="Ấn ESC để thoát",fg="black",bd=0,bg="#DDDDDA" )
    lable1.config(font=("",10))
    lable1.pack(pady=0)
    
    
    ND = tkinter.Button(button_frame,text="Nhận Diện",font=(("Arial"),10,'bold'),bg="#303030",fg="#FFFFFF",command=nhandien )
    ND.pack(pady=10)
    
    vatthe = tkinter.Label(root,text="",fg="black",bd=0,bg="#FFFFFF")
    vatthe.config(font=("",10))
    vatthe.pack(pady=0)
    vatthe.pack(padx=0)
    
    acc = tkinter.Label(root,text="",fg="black",bd=0,bg="#FFFFFF")
    acc.config(font=("",10))
    acc.pack(pady=0)
    acc.pack(padx=0)
    
    root.mainloop()

giaodien()
camera = cv2.VideoCapture(0)
