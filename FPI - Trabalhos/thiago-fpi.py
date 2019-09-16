

import tkinter as tk
from PIL import Image
import cv2

class Application(tk.Frame):

    img_gramado_22 = 'Gramado_22k.jpg'
    img_gramado_72 = 'Gramado_72k.jpg'
    img_space_187 = 'Space_187k.jpg'
    img_space_46 = 'Space_46k.jpg'
    img_underwater_53 = 'Underwater_53k.jpg'

    curr_img = None

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
        self.cria_botao()

    def cria_botao(self):
        self.curr_img = tk.StringVar()
        self.curr_img.set(self.img_gramado_22)
        for name, path in self.MODES:
            tk.Radiobutton(root, text=name, variable=self.curr_img, value=path).pack(anchor=tk.W)

        self.btn_1 = tk.Button(self)
        self.btn_1["text"] = "Imagem original"
        self.btn_1["command"] = self.originalimg
        self.btn_1.pack(side="right")

        self.btn_2 = tk.Button(self)
        self.btn_2["text"] = "Escala de cinza"
        self.btn_2["command"] = self.cinzaimg
        self.btn_2.pack(side="right")

        self.btn_3 = tk.Button(self)
        self.btn_3["text"] = "Rotação"
        self.btn_3["command"] = self.rotacaoimg
        self.btn_3.pack(side="left")

        self.btn_4 = tk.Button(self)
        self.btn_4["text"] = "Espelhada na Vertical"
        self.btn_4["command"] = self.flipverticalimg
        self.btn_4.pack(side="left")

        self.btn_5 = tk.Button(self)
        self.btn_5["text"] = "Espelhada na Horizontal"
        self.btn_5["command"] = self.fliphorizontalimg
        self.btn_5.pack(side="left")

        self.btn_6 = tk.Button(self)
        self.btn_6["text"] = "Mostra Imagem Quantizada"
        self.btn_6["command"] = self.quantizadaimg
        self.btn_6.pack(side="left")

        # self.btn_7 = tk.Button(self)
        # self.btn_7["text"] = "Mostra Imagem Quantizada"
        # self.btn_7["command"] = self.d_hmqimg
        # self.btn_7.pack(side="left")


        self.quit = tk.Button(self, text="Sair", fg="black",
                              command=root.destroy)
        self.quit.pack(side="bottom")



    # # Mostra a imagem Original
    # def originalimg (self):
    #     path = self.curr_img.get()
    #     img = cv2.imread(path)
    #     cv2.imshow('Original', img)
    #     print('Imagem original - ok ')

    #
    # # Mostra imagem escala de cinza
    # def cinzaimg (self):
    #     path = self.curr_img.get()
    #     img = cv2.imread(path)
    #     gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     cv2.imshow('Escala de cinza', gray_image)
    #     print('Imagem Escala de cinza - ok .')

    #
    # # Mostra imagem rotacionada
    # def rotacaoimg (self):
    #     path = self.curr_img.get()
    #     img = cv2.imread(path)
    #     rows, cols = img.shape[:2]
    #
    #     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    #     dst = cv2.warpAffine(img, M,(cols, rows))
    #     cv2.imshow('Rotated Image', dst)
    #     print('Rotated Image loaded.')


    # # Display flipped image (vertical)
    # def d_fvimg(self):
    #     path = self.curr_img.get()
    #     img = cv2.imread(path)
    #
    #     v_img = img.copy()
    #     if self.vf:
    #         v_img = img
    #         self.vf = False
    #     else:
    #         v_img = cv2.flip( img, 1)
    #         self.vf = True
    #
    #     cv2.imshow( "Vertical flip", v_img )
    #     print('Vertical Flipped Image loaded.')
    #     cv2.imwrite('v-flipped-img.jpeg', v_img)
    #     print('Vertical Flipped Image saved.')


    # # Display flipped image (horizontal)
    # def d_fhimg(self):
    #     path = self.curr_img.get()
    #     img = cv2.imread(path)
    #
    #     h_img = img.copy()
    #     if self.hf:
    #         h_img = img
    #         self.hf = False
    #     else:
    #         h_img = cv2.flip( img, 0 )
    #         self.hf = True
    #
    #     cv2.imshow( "Horizontal flip", h_img )
    #     print('Horizontal Flipped Image loaded.')
    #     cv2.imwrite('h-flipped-img.jpeg', h_img)
    #     print('Horizontal Flipped Image saved.')


    # Display quantized image
    def d_qimg(self):
        path = self.curr_img.get()
        img = Image.open(path)
        img = img.quantize(16)

        img.convert('RGB').save("q-16-img.jpeg")
        print('Quantized 16x Image saved.')

        path = 'q-16-img.jpeg'
        img = cv2.imread(path)

        cv2.imshow( "Quantized 16x", img )
        print('Quantized 16x Image loaded.')


    def d_hmqimg(self):

        path = self.curr_img.get()
        img = cv2.imread(path)
        width, height = img.shape[:2]

        q = 32
        r = 256 / q # Ensure your image is on [0 255] range

        for row in range(0, height):
            for col in range(0, width):
                img[col][row] = abs(img[col][row] / r)

        cv2.imshow( "Handmade Quantized 16x", img )
        print('Handmade Quantized 16x Image loaded.')


root = tk.Tk()
app = Application(master=root)
app.mainloop()
