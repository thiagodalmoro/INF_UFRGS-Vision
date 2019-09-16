#
# Author: Pedro Cantarutti
#

import tkinter as tk
from PIL import Image
import cv2

class Application(tk.Frame):

    img_gramado_22 = 'test_images/Gramado_22k.jpg'
    img_gramado_72 = 'test_images/Gramado_72k.jpg'
    img_space_187 = 'test_images/Space_187k.jpg'
    img_space_46 = 'test_images/Space_46k.jpg'
    img_underwater_53 = 'test_images/Underwater_53k.jpg'

    curr_img = None
    gs_img = None

    MODES = [
        ("Gramado22", img_gramado_22),
        ("Gramado72", img_gramado_72),
        ("Space187", img_space_187),
        ("Space46", img_space_46),
        ("Underwater53", img_underwater_53)
    ]

    vf = False
    hf = False

    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.curr_img = tk.StringVar()
        self.curr_img.set(self.img_gramado_22)
        for name, path in self.MODES:
            tk.Radiobutton(root, text=name, variable=self.curr_img, value=path).pack(anchor=tk.W)

        self.btn_1 = tk.Button(self)
        self.btn_1["text"] = "Mostra Imagem original"
        self.btn_1["command"] = self.d_oimg
        self.btn_1.pack(side="top")

        self.btn_2 = tk.Button(self)
        self.btn_2["text"] = "Mostra Imagem em escala de cinza"
        self.btn_2["command"] = self.d_gimg
        self.btn_2.pack(side="top")

        self.btn_3 = tk.Button(self)
        self.btn_3["text"] = "Mostra Imagem rotacionada."
        self.btn_3["command"] = self.d_rimg
        self.btn_3.pack(side="left")

        self.btn_4 = tk.Button(self)
        self.btn_4["text"] = "Mostra Imagem Espelhada na Vertical"
        self.btn_4["command"] = self.d_fvimg
        self.btn_4.pack(side="left")

        self.btn_5 = tk.Button(self)
        self.btn_5["text"] = "Mostra Imagem Espelhada na Horizontal"
        self.btn_5["command"] = self.d_fhimg
        self.btn_5.pack(side="left")

        self.btn_6 = tk.Button(self)
        self.btn_6["text"] = "Mostra Imagem Quantizada"
        self.btn_6["command"] = self.d_qimg
        self.btn_6.pack(side="left")

        self.btn_7 = tk.Button(self)
        self.btn_7["text"] = "Mostra Imagem Quantizada"
        self.btn_7["command"] = self.d_hmqimg
        self.btn_7.pack(side="left")


        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=root.destroy)
        self.quit.pack(side="bottom")



    # Display original image
    def d_oimg(self):
        path = self.curr_img.get()
        img = cv2.imread(path)
        cv2.imshow('original', img)
        print('Original Image loaded.')


    def avg(self, p):
        return (0.299*p[0] + 0.587*p[1] + 0.114*p[2]) / 3


    # Display gray scale image
    def d_gimg(self):
        if self.gs_img is None:
            path = self.curr_img.get()
            img = cv2.imread(path)
            for row in range(len(img)):
                for col in range(len(img[row])):
                    img[row][col] = self.avg(img[row][col]);

            self.gs_img = img
        else:
            for row in range(len(self.gs_img)):
                for col in range(len(self.gs_img[row])):
                    self.gs_img[row][col] = self.avg(self.gs_img[row][col]);

        cv2.imshow('Gray Image', self.gs_img)
        print('Gray Image loaded.')


    # Display rotated image
    def d_rimg(self):
        path = self.curr_img.get()
        img = cv2.imread(path)
        rows, cols = img.shape[:2]

        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        dst = cv2.warpAffine(img, M,(cols, rows))
        cv2.imshow('Rotated Image', dst)
        print('Rotated Image loaded.')


    # Display flipped image (vertical)
    def d_fvimg(self):
        path = self.curr_img.get()
        img = cv2.imread(path)

        v_img = img.copy()
        if self.vf:
            v_img = img
            self.vf = False
        else:
            v_img = cv2.flip( img, 1)
            self.vf = True

        cv2.imshow( "Vertical flip", v_img )
        print('Vertical Flipped Image loaded.')
        cv2.imwrite('v-flipped-img.jpeg', v_img)
        print('Vertical Flipped Image saved.')


    # Display flipped image (horizontal)
    def d_fhimg(self):
        path = self.curr_img.get()
        img = cv2.imread(path)

        h_img = img.copy()
        if self.hf:
            h_img = img
            self.hf = False
        else:
            h_img = cv2.flip( img, 0 )
            self.hf = True

        cv2.imshow( "Horizontal flip", h_img )
        print('Horizontal Flipped Image loaded.')
        cv2.imwrite('h-flipped-img.jpeg', h_img)
        print('Horizontal Flipped Image saved.')


    # Display quantized image.
    def d_qimg(self):
        path = self.curr_img.get()
        img = Image.open(path)
        img = Image.quantize(16)

        Image.convert('RGB').save("q-16-img.jpeg")
        print('Quantized 16x Image saved.')

        path = 'q-16-img.jpeg'
        img = cv2.imread(path)

        cv2.imshow( "Quantized 16x", img )
        print('Quantized 16x Image loaded.')


    def d_hmqimg(self):
        # TODO(pac):
        # - Check algorithm
        path = self.curr_img.get()
        img = cv2.imread(path)
        width, height = img.shape[:2]

        q = 32
        r = 256 / q # Ensure your image is on [0 255] range

        for row in range(0, height):
            for col in range(0, width):
                img[col][row] = abs(img[col][row] / r)

        cv2.imshow( "Handmade Quantized 32x", img )
        print('Handmade Quantized 32x Image loaded.')



root = tk.Tk()
app = Application(master=root)
app.mainloop()
