#!/bin/bash

# Send image.png to the YOLO prediction endpoint
curl -X POST http://localhost:8667/predict \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.png"
