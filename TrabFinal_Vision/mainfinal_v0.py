import cv2


#
#
# # #imagem 1 - Original
# img1 = cv.imread('figure1.jpg',1)
#
# gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
#
# cv.imshow('Image1', img1)
# cv.imshow ('Image Gray', gray)
#
# start_pointx = (479, 9)
# end_pointx = (450, 536)
#
# #color = BLUE (B, G, R)
# color = (255,0,0)
#
# #pxl px
# thinckness = 3
#
# # # Draw a diagonal blue line with thickness of 5 px
# img = cv.line(img1,start_pointx ,end_pointx,color,thinckness)
# cv.imshow('ImageDraw', img)
#
#
# key= cv.waitKey(0)
# if key == 27:         # wait for ESC key to exit
#     print('Key ESC pressed.')
#     cv.destroyAllWindows()



def main():

    cap = cv2.VideoCapture("CTL.mp4")
    cv2.namedWindow('Frame')

    start_pointx = (479, 9)
    end_pointx = (450, 536)

    # color = BLUE (B, G, R)
    color = (255, 0, 0)

    # pxl px
    thinckness = 3


    while(True):

        ret, frame = cap.read()
            if not ret:
              break




        key= cv2.waitKey(0)
        if key == 27:         # wait for ESC key to exit
            print('Key ESC pressed.')
            break

        # cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
   main()



