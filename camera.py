import cv2
import boto3
import datetime
import requests

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor = 0.6

count = 0


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        # count=0
        global count
        success, image = self.video.read()
        is_success, im_buf_arr = cv2.imencode(".jpg", image)
        image1 = im_buf_arr.tobytes()
        client = boto3.client('rekognition',
                              aws_access_key_id="ASIAS3CQ4KIYBVOLQU5P",
                              aws_secret_access_key="Slp5ri99RUTbkRiFBTWDgpNQhgDguUKvXEZib5sB",
                              aws_session_token="FwoGZXIvYXdzEMv//////////wEaDCt6Phl1Xs8T8R9UUiLGATTjqxcVJ8qkJWr+31Ujb3jrWiuPcQELnEPPhhxM0lskzLaSTMs6aYeBFEK9AUdhhOlyTakzMkIpKsux3CQYAumt/i/BBZXbmd4qIuKM+0gktIf36n4v1RwAVWUDxwEGPnylztOfYxVusuQzKYLtdtYu2q4zWbhqyK1As6DNXeN0kU0xcK6MOAemycrFT4vjiY0r96+Cr0sA6FukZYwfP9BAXZwhg2l0zWYA6ZwRe47hTtf48EEIVudVORcTJHBina7mZQiHtSj20/76BTIttj6lQIV6xgEYQqq2POcB1gfUOovtF2oP0Efj7R3SKNDdGUj3f86YLKPsUc89",
                              region_name='us-east-1')
        response = client.detect_custom_labels(
            ProjectVersionArn='arn:aws:rekognition:us-east-1:195590640176:project/MaskDetection/version/MaskDetection.2020-09-10T20.25.25/1599749726067',
            Image={
                'Bytes': image1})
        print(response['CustomLabels'])

        if not len(response['CustomLabels']):
            count = count + 1
            date = str(datetime.datetime.now()).split(" ")[0]
            # print(date)
            url = "https://bd05xb9zkf.execute-api.us-east-1.amazonaws.com/Main123" + date + "&count=" + str(count)
            resp = requests.get(url)
            f = open("countfile.txt", "w")
            f.write(str(count))
            f.close()
            # print(count)

        image = cv2.resize(image, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in face_rects:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break
        ret, jpeg = cv2.imencode('.jpg', image)
        # cv2.putText(image, text = str(count), org=(10,40), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(1,0,0))
        cv2.imshow('image', image)
        return jpeg.tobytes()