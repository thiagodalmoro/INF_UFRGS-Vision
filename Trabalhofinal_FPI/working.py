# Trabalho 03 - FPI
#
# Thiago Dal Moro
#
#
#criação de videos e frames
#

import numpy as np
import cv2
import math
from tkinter import X
from tkinter.ttk import Style
import PIL.Image, PIL.ImageTk
from PIL import Image, ImageEnhance
import tkinter as tk



#bibliotecas

# window = tk.Tk()
# window.title("Trabalho FPI - 03 - Thiago Dal Moro")
# window.geometry ("340x460")
#


def switches(switch):
    cv2.createTrackbar(switch, 'Video Frame', 0, 1, app_cota)
    cv2.createTrackbar('lower', 'Video Frame', 0, 255, app_cota)
    cv2.createTrackbar('upper', 'Video Frame', 0, 255, app_cota)
    cv2.createTrackbar('GaussianBlur', 'Video Frame', 3, 5, app_cota)

def rotate(frame,angle):

    altura, largura = frame.shape[:2]
    ponto = (largura / 2, altura / 2) #ponto no centro da figura

    rotacao = cv2.getRotationMatrix2D(ponto, angle,1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)

    bw = int((altura * abs(sin)) + (largura * abs(cos)))
    bh = int((altura * abs(cos)) + (largura * abs(sin)))

    rotacao[0, 2] += ((bw / 2) - ponto[0])
    rotacao[1, 2] += ((bh / 2) - ponto[1])

    rotacionado = cv2.warpAffine(frame, rotacao,(bw, bh))

    return rotacionado

def rescale (frame, percent):

    frame_width = int(frame.shape[1] * percent/ 100)
    frame_height = int(frame.shape[0] * percent/ 100)
    tamanho = (frame_width, frame_height)
    frame_rescale = cv2.resize(frame, tamanho, interpolation=cv2.INTER_AREA)

    return frame_rescale


def flip_frame (frame, code):
    flipped = cv2.flip(frame, flipCode=code)
    return flipped

    #Função que cria a frames em tons de Cinza
def cinza(frame):

     gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     return gray_image

def gaussian_blur (frame):
    blur = cv2.GaussianBlur(frame,(3,3),0)
    return blur

def detec_sin (frame, lower, upper, sinal):
    if sinal == 0:
        frame_alter = frame
    else:
        frame_alter = cv2.Canny(frame, lower, upper)
    return frame_alter


def main():

    cap = cv2.VideoCapture(0)

    #gravacao do video
    #criacao do arquivo de video
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 15.0, (int(frame_width),int(frame_height)))

    cv2.namedWindow('Video Frame')

    switch = ' 0 : OFF \n    1 : ON'

    switches(switch)

    while(True):
        ret, frame = cap.read()
        if not ret:
            break
        #trackbars
        lower = cv2.getTrackbarPos('lower', 'Video Frame')
        upper = cv2.getTrackbarPos('upper', 'Video Frame')
        sinal = cv2.getTrackbarPos(switch, 'Video Frame')


        #funcoes
        edge  = detec_sin (frame, lower, upper, sinal) #caputra edges
        cv2.imshow('Canny', edge)

        gaussian = gaussian_blur(frame) #filtro gaussiano
        cv2.imshow('Gaussianblur', gaussian)

        gray_sclae = cinza(frame) # escala de cinza
        cv2.imshow('Escala de Cinza', gray_sclae)

        rotaciona = rotate(frame,90)
        cv2.imshow('rotacionado', rotaciona)

        frame_rescale = rescale (frame, 50)
        cv2.imshow('Rescale', frame_rescale)

        flipped_v = flip_frame (frame, 1) #vertical
        cv2.imshow('flip v', flipped_v)
        flipped_h = flip_frame (frame, 0) #horizontal
        cv2.imshow('flip h', flipped_h)

        sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
        sobelx_64 = np.absolute(sobelx)
        sobelx_8u = np.uint8(sobelx_64)
        cv2.imshow('SobelX', sobelx_8u)

        sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
        sobely_64 = np.absolute(sobely)
        sobely_8u = np.uint8(sobely_64)
        cv2.imshow('SobelY', sobely_8u)

    # imagens mostradas
        cv2.imshow('Original', frame)

    # videos mostrados
        cv2.imshow('Video Frame', edge)

    # condição para saida do programa
        k = cv2.waitKey(1)
        if k == 27:         # wait for ESC key to exit
            print('Key ESC pressed.')
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def app_cota (value):
    #
    # if (lower != 0)
        print('Valor:', value)

if __name__ == '__main__':
   main()
