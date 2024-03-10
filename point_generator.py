import cv2
import numpy as np
from numpy import asarray
import math,random,os

#some parameters
canvas = np.zeros((600,600,3), dtype ="uint8")
canvas.fill(255)
radius = 1
radius2 = 1
line_thikness=1
line_darkness=255
color = (22, 22, 144)
desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
if os.path.isfile(desktop+'/image.jpg'):
    img = cv2.imread(desktop+'/image.jpg')
else:
    file = 'image.jpg'
    open(file, 'a').close()
img = asarray(img)
img = cv2.resize(img,(600,600),interpolation = cv2.INTER_AREA)
img = cv2.UMat(img)
output_size=(600,600)
line_dest=45
point_count=150
point2_count=150
point_r=1
point_g=1
point_b=1
point2_r=1
point2_g=1
point2_b=1
pt=[]
speed2=3

def point_ran_gen():
    for i in range(0, 1200):
        item = (np.random.randint(0, high=600, size=(3,)))
        speed = random.randint(1, 5)
        item2 = item[0], item[1],math.cos(item[2]), math.sin(item[2]),speed/5
        pt.append(item2)
        print(item2)
    #for x in pt:
    #    cv2.circle(canvas, tuple(x), radius, color, -1)


def point_ran_mov():

    movv=-1
    for x in pt:
        movv=movv+1
        # new_point=int(x[0]+math.cos(x[2])),int(x[1]/100+math.sin(x[2]))
        pt[movv]=x[0]+(x[2]*x[4]*speed2),x[1]+(x[3]*x[4]*speed2),x[2],x[3],x[4]
        if x[0] > 600:
            pt[movv] = x[0]-600, x[1],x[2],x[3],x[4]
        elif x[0] < 0:
            pt[movv] = x[0]+600, x[1],x[2],x[3],x[4]
        if x[1] > 600:
            pt[movv] = x[0],x[1]-600,x[2],x[3],x[4]
        elif x[1] < 0:
            pt[movv] = x[0],x[1]+600,x[2],x[3],x[4]
        new_point=int(x[0]+(x[2]*x[4]*speed2)),int(x[1]+(x[3]*x[4]*speed2))
        if movv < point_count:
            new_point_list.append(new_point)
            #print(movv,new_point, x[0], x[1], x[2],x[3],x[4])
            cv2.circle(canvas, tuple(new_point), radius, (point_b,point_g,point_r), -1)
        elif movv < point2_count+point_count:
            cv2.circle(canvas, tuple(new_point), radius2, (point2_b,point2_g,point2_r), -1)


def generate_lines():
    for x in new_point_list:
        for i in new_point_list:
            if x[0] < (i[0]+line_dest-1) and x[0] > (i[0]-line_dest+1) and x[1] < (i[1]+line_dest-1) and x[1] > (i[1]-line_dest+1):
                cv2.line(canvas, x, i, (line_darkness, line_darkness, line_darkness), line_thikness)
                #points=x,i
                #new_list_v.append(points)
                #print(i)

#fourcc = cv2.cv.CV_FOURCC(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#out = cv2.VideoWriter('output.avi', fourcc, 50, output_size)
out=cv2.VideoWriter(desktop+'\output.mp4',cv2.VideoWriter_fourcc('X','V','i','D'), 50, (600,600))


point_ran_gen()
pt = cv2.UMat(np.array(pt, dtype=np.uint8))
new_point_list=[]
new_list_v=[]



#making bars and buttons to use
def nothing(x):
    #print(x)
    pass

cv2.namedWindow('Bars')
cv2.resizeWindow('Bars',500,700)
cv2.createTrackbar('speed','Bars',1,40,nothing)
cv2.createTrackbar('line_dest','Bars',30,600,nothing)
cv2.createTrackbar('line_thikness','Bars',0,20,nothing)
cv2.createTrackbar('line_darkness','Bars',0,255,nothing)
cv2.createTrackbar('point_count','Bars',150,600,nothing)
cv2.createTrackbar('point_size','Bars',3,20,nothing)
cv2.createTrackbar('point_r','Bars',3,255,nothing)
cv2.createTrackbar('point_g','Bars',3,255,nothing)
cv2.createTrackbar('point_b','Bars',3,255,nothing)
cv2.createTrackbar('point2_count','Bars',150,600,nothing)
cv2.createTrackbar('point2_size','Bars',1,20,nothing)
cv2.createTrackbar('point2_r','Bars',3,255,nothing)
cv2.createTrackbar('point2_g','Bars',3,255,nothing)
cv2.createTrackbar('point2_b','Bars',150,255,nothing)
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar('switch', 'Bars',0,1,nothing)
record = '0 : OFF \n1 : ON'
cv2.createTrackbar('record', 'Bars',0,1,nothing)


while True: #main loop

    switch = cv2.getTrackbarPos('switch', 'Bars')
    if switch == 0:
        canvas = cv2.copyTo(img, img)
    else:
        canvas.fill(255)

    point_ran_mov()
    generate_lines()
    new_point_list = [] #empyy the list

    #laout retrive values
    speed2 = cv2.getTrackbarPos('speed', 'Bars')
    line_dest=cv2.getTrackbarPos('line_dest','Bars')
    line_darkness=cv2.getTrackbarPos('line_darkness','Bars')
    line_thikness=cv2.getTrackbarPos('line_thikness','Bars')+1
    point_count=cv2.getTrackbarPos('point_count','Bars')
    radius = cv2.getTrackbarPos('point_size', 'Bars')
    point_r = cv2.getTrackbarPos('point_r', 'Bars')
    point_g = cv2.getTrackbarPos('point_g', 'Bars')
    point_b = cv2.getTrackbarPos('point_b', 'Bars')
    point2_count=cv2.getTrackbarPos('point2_count','Bars')
    radius2=cv2.getTrackbarPos('point2_size','Bars')
    point2_r = cv2.getTrackbarPos('point2_r', 'Bars')
    point2_g = cv2.getTrackbarPos('point2_g', 'Bars')
    point2_b = cv2.getTrackbarPos('point2_b', 'Bars')

    frame = cv2.resize(canvas,output_size,interpolation = cv2.INTER_AREA)

    # write the  frame
    record=cv2.getTrackbarPos('record','Bars')
    if record == 0:
        cv2.putText(canvas,"Not recording",(0, 30),cv2.FONT_HERSHEY_SIMPLEX,1,(209, 80, 0, 255),3)
    else:
        out.write(frame)

    k = cv2.waitKey(1)
    if k!= -1:
        break
    cv2.imshow("Simulation", canvas)

out.release()
cv2.destroyAllWindows()
