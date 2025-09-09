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



**Tips**

1. Always capture dark frames and reference (blank water) frames for each measurement to cancel ambient and lamp drift.

2. Use band-pass emission filters for fluorescence imaging (blocks excitation light).

3. Use short pulsed illumination and synchronized capture to increase SNR.

4. Create a controlled training dataset: multiple polymer types, sizes, pigments, and aged/biofouled samples. Real-world noise is the killer.

5. For ML labels, pair your multispectral+camera readings with corresponding FTIR/Raman lab-confirmed labels.



**To-Do**

1. Automatic Hardware Checking

Install a flow sensor before the chamber and compare actual vs expected flow to detect leaks or blockages. Place a reference LED–photodiode pair inside to measure baseline light transmission, and run a self-test by shining the LED onto a white/PTFE patch for calibration. ESP32-S3 runs a loop comparing these signals to thresholds; if deviation persists, it raises a hardware error and sends an app alert.

2. Auto-Cleaning with Ionized Water

Use a 3-way solenoid valve to switch between sample water and a cleaning tank. During cleaning mode, a pump flushes the chamber with ionized or UV-treated water (produced via electrolysis plates or stored cartridge) while a UV-C LED sterilizes the flow path. After flushing, the system records a blank baseline to reset measurements automatically.

3. Plastic Concentration Alert System

Measure scattering + absorption features and feed them into a TinyML model to estimate plastic concentration (ppm). If values exceed a set threshold, ESP32-S3 sends data via Wi-Fi/BLE to Firebase/InfluxDB, which triggers a push notification on the mobile app. The app shows real-time concentration graphs, alerts, and historical trends for monitoring.



**Must Include References***(Given Datasets)*

1. Stressor-AOP : https://cb.imsc.res.in/saopadditives/
2. NOAA NCEI Marine DB : https://pmc.ncbi.nlm.nih.gov/articles/PMC10589325/
3. Marine Debris Program : https://marinedebris.noaa.gov/


