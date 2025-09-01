**Steps**

1. Pass water through a narrow ledge(preprocess if needed)
2. Shine controlled light(blue/UV) on it
3. Read the light using small spectrometer + camera
4. Analyse plastic or not



**Working**

1. Water enters *dark chamber*
2. Filtering
3. Light is passed
4. Stain is formed as a spectrum
5. Spectrum is captured using AS7265x and image is captured using ESP32-S3
6. Wi-Fi/Bluetooth is used to stream data on mobile app



**Arrangement**

>A box painted full black inside(dark chamber)

>Water flow path is at centre

>Lighting is on one side

>Multispectral sensor can be placed at opposite(for on-axis imaging) or at 90°(for scattering imaging){accuracy/feasibility poruthu yosipom}

>Remaining space la wiring, battery, filters can be placed



**Architecture**

1. ML model : 1D-CNN(preferred) or Random Forest

{

Written in : TensorFlow

Converted to : TfLite

Reason => TinyML : For small, low power devices

}

2\. HW : 

{

ESP32-S3 : Cheap and has WiFi

AS7341 OR AS7265x : Multispectral sensors

A box : literally just a box :)

}

3\. TechStack :

{

FIRMWARE : <above>

Backend : Python

Frontend : Dart(flutter) as mobile app is more reliable

DB : 

Prototype - FireBase [Free tier is large and simple language]

Final Project - InfluxDB [Better for time series]

}



**Problems**

1. Camera : Low cost camera can't magnify 10-50µm
2. Nile Red : Will cause false positives
3. Battery : Long, safe and efficient use
4. Data : Real samples are noisy, training data should be usable and real
5. Plastic class : Use Fourier Transform Infrared Spectrometry(FTIR)\[MOST IMPORTANT STEP]



**Cost**

ESP32-S3 : ~2000

AS7265x : ~9000

UV LED torch (365–395 nm) : ~1000

Battery Li-ion(12v 5Ah) : ~1500

=>

*Decent specs = ~17k*

**Must Include References***(Given Datasets)*

1. Stressor-AOP : https://cb.imsc.res.in/saopadditives/
2. NOAA NCEI Marine DB : https://pmc.ncbi.nlm.nih.gov/articles/PMC10589325/
3. Marine Debris Program : https://marinedebris.noaa.gov/


