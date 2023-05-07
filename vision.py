import numpy as np
import cv2
from cvzone.ColorModule import ColorFinder
import cvzone
import torch
import os
from PIL import Image, ImageDraw, ImageFont
from Machine_Learning.architecture import Network



class Kanji_Guesser:
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.model = Network(len_kanji_dict=self.num_samples)
        self.model.load_state_dict(torch.load("Machine_Learning/model.pth"))
        self.kanji_dict = self.init_kanji_dict()
        
    def parse_prediction(self, x, top):
        return [self.kanji_dict[pred] for pred in torch.topk(x, top).indices.squeeze().numpy()]
        
    def predict(self, x):
        return self.model(x)
    
    def init_kanji_dict(self):
        kanjis = os.listdir('Machine_Learning/data')[:self.num_samples]
        kanjis = list(map(lambda x: x[:-4], kanjis))
        kanji_dict = { i: kanji for (i, kanji) in enumerate(kanjis) }
        return kanji_dict


kanji_guesser = Kanji_Guesser(num_samples=100)



cap = cv2.VideoCapture(2)

color_finder = ColorFinder(trackBar=False)
mask_vals = {'hmin': 164, 'smin': 101, 'vmin': 215, 'hmax': 179, 'smax': 255, 'vmax': 255}


if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame")
        continue

    masked, mask = color_finder.update(frame, mask_vals)
    ret, thresh1 = cv2.threshold(masked, 0, 255, cv2.THRESH_BINARY)
    eroded = cv2.erode(thresh1, (4,4), iterations=5)
    dilated = cv2.dilate(eroded, (4,4), iterations=15)
    gray = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)
    
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 40]
    if contours:
        centers = []
        for i in contours:
            M = cv2.moments(i)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                centers.append((cx, cy))
                
        if len(contours) == 4:
            pts1 = np.float32(centers)
            pts1[[0, 1], :] = pts1[[1, 0], :]
            pts1[[2, 3], :] = pts1[[3, 2], :]
            
            whiteboard_aspect_ratio = (4,3)
            whiteboard_scale_factor = 200
            
            pts2 = np.float32([
                            [0,whiteboard_aspect_ratio[1]*whiteboard_scale_factor],
                            [0,0],
                            [whiteboard_aspect_ratio[0]*whiteboard_scale_factor,whiteboard_aspect_ratio[1]*whiteboard_scale_factor],
                            [whiteboard_aspect_ratio[0]*whiteboard_scale_factor,0],
            ])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            warped = cv2.warpPerspective(frame, matrix, (whiteboard_aspect_ratio[0]*whiteboard_scale_factor,whiteboard_aspect_ratio[1]*whiteboard_scale_factor))
            
            # Warped image of whiteboard 
            whiteboard = warped
            
            margin = 75
            cropped = whiteboard[margin:whiteboard.shape[0]-margin, margin:whiteboard.shape[1]-margin]
            
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            thresh = cv2.bitwise_not(thresh)
            
            kanji_contours, kanji_hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            try:
                rects = [cv2.boundingRect(c) for c in kanji_contours]
                x = min(list(map(lambda x: x[0], rects)))
                y = min(list(map(lambda x: x[1], rects)))
                w = max(list(map(lambda x: x[0] + x[2], rects)))
                h = max(list(map(lambda x: x[1] + x[3], rects)))
                
                char_margin = int((w - x) / 7)
                character = thresh[y-char_margin:h+char_margin, x-char_margin:w+char_margin]
                character = cv2.resize(character, (64, 64), interpolation=cv2.INTER_AREA)
                # character = cv2.dilate(character, (5,1), iterations=1)
                
                tensor_char = torch.FloatTensor(character).unsqueeze(0).unsqueeze(0)
                tensor_char = tensor_char.apply_(lambda x: 1 if x > 100 else -1)
                output = kanji_guesser.predict(tensor_char)
                prediction = kanji_guesser.parse_prediction(output, top=5)
                
            except:
                pass
            
            try:
                cv2.imshow('warped', character)
            except:
                pass
            
        if centers: 
            for c in centers:
                frame = cv2.circle(frame, c, 10, (255,0,0), -1)
        
        try:
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            draw.rectangle((0,0,3000,220), fill=(0,0,0))
            font = ImageFont.truetype("MSMINCHO.TTF", size=200)
            draw.text((10, 10), ' '.join(prediction[0]), font=font)
            font = ImageFont.truetype("MSMINCHO.TTF", size=100)
            draw.text((260, 10), ' '.join(prediction[1:]), font=font)
            frame = np.array(img_pil)
                
        except NameError:
            print('no prediction')
            
        cv2.imshow('frame', frame)
    

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
