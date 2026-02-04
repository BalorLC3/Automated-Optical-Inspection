## What is Automated Optical Inspection (AOI)?
In industries there is a continuous need of automating tasks and continuous improvement. Detect failures to ensure quality is a must do in any industry. In this project I designed a system which ingests data and detects the type of failure in metal sheets

* Inference of 120ms
* Low latency
* Good predictions

Here I used YOLO26 (the most recent version of YOLO) as the computer vision algorithm, which got a high F1-score of 80% after low training effort

## Structure
The backend is written in Go, then Go sends data to Python where inference is made and lastly the user can get the output in a small SCADA like web-page written in react.

## Demo
1. Ingest the system with a image
2. Get a response

| <img src="docs/defect-detection.png" width="500" /> |