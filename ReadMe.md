# Microplastic Detector — README

**Steps**

1. Pass water through a narrow ledge (preprocess if needed)
2. Shine controlled light (1 W white LED) on it
3. Read the light using small spectrometer + camera
4. Analyse: *plastic or not*

**Working**

1. Water enters **dark chamber**
2. Filtering (membrane or cuvette)
3. Light is passed (controlled, repeatable illumination)
4. Stain or optical response is formed as a spectrum / image
5. **Spectrum is captured using AS7265x** and image is captured using **ESP32-CAM (OV2640)**
6. **Wi‑Fi** is used to stream data to a mobile app / backend

**Arrangement**

> A box painted full black inside (*dark chamber*)

> Water flow path is at centre

> Lighting is on one side (choose geometry: on‑axis or 90° scattering)

> **Multispectral sensor (AS7265x)** can be placed at opposite (for **on‑axis transmission**) or at **90°** (for scattering/size info) — *evaluate accuracy & feasibility*.

> Remaining space: wiring, battery, filters, mounting hardware

**Architecture**

1. **ML model** : *CNN*

   * **Written in:** *TensorFlow*
   * **Converted to:** *TFLite*
   * **Reason:** TinyML for small, low‑power devices (on‑device inference or mobile inference)

2. **HW** :

   * **ESP32‑S3 (N16R8 recommended)** — Wi‑Fi, PSRAM for camera buffering and TinyML
   * **AS7265x** — 18‑channel multispectral sensor (410–940 nm) *(this project uses AS7265x only)*
   * **ESP32‑CAM (OV2640)** camera module (or OV2640 mounted on compatible board)
   * **1 W white LED** (controlled illumination)
   * **A box** — dark enclosure for optics

3. **TechStack** :

   * **FIRMWARE:** C++ (ESP‑IDF / Arduino) on ESP32‑S3
   * **Backend:** Python (data store, heavier models, analytics)
   * **Frontend:** Dart / Flutter (mobile app for field use)
   * **DB:** Prototype — *Firebase* (easy, free tier); Production — *InfluxDB* (time series optimized)

**Problems**

1. **Camera:** low‑cost camera without magnification cannot reliably resolve **10–50 µm** — magnification or microscope objective required
2. **Nile Red staining:** useful for fluorescence screening but can cause false positives with organics. Note: fluorescence workflows typically require UV or narrowband excitation and optical filtering. Using a white LED limits fluorescence sensitivity.
3. **Battery:** ensure safe, long runtime (BMS, fusing, regulators)
4. **Data:** field samples are noisy — collect diverse, real labeled data early
5. **Plastic class:** FTIR/Raman remains the gold standard for polymer confirmation (use for validation)

**Cost (approx.)**

* ESP32‑S3 : \~₹2,000
* AS7265x : \~₹9,000
* 1W white LED : \~₹1,000
* Battery Li‑ion (12 V, 5 Ah) : \~₹1,500

**=> Decent specs ≈ ₹13,500**

**Tips**

1. Always capture **dark frames** and **reference (blank water)** frames for every measurement to cancel ambient and lamp drift.
2. Use **band‑pass / long‑pass filters** if you run fluorescence (blocks excitation). Not required for reflectance workflow.
3. Use **short pulsed illumination** synced with capture to increase SNR and reduce heat.
4. Create a **controlled training dataset**: multiple polymer types, sizes, pigments, and aged/biofouled samples. Real‑world noise reduces model performance if not represented.
5. For ML labels, pair your multispectral + camera readings with FTIR/Raman validated labels whenever possible.

**To‑Do (feature list / roadmap)**

1. **Automatic Hardware Checking**

   * Install a flow sensor before the chamber to detect leaks/blockages.
   * Add a reference LED–photodiode pair to measure baseline light transmission.
   * ESP32 runs a self‑test loop (white patch → measure → compare thresholds). If persistent deviation, raise hardware error and send alert to app.

2. **Auto‑Cleaning with Ionized Water (optional advanced)**

   * Use a 3‑way solenoid valve to switch between sample and cleaning water.
   * Pump flushes chamber while a UV‑C LED sterilizes the flow path.
   * Record blank baseline after cleaning to reset measurements.

3. **Plastic Concentration Alert System**

   * Measure scattering + absorption features → feed to TinyML model to estimate concentration (ppm).
   * If threshold exceeded, ESP32 sends data to Firebase/InfluxDB → triggers push notification.
   * Mobile app displays real‑time concentration graphs, alerts, and histories.

**What I used (current hardware inventory)**

* **ESP32‑S3 N16R8 (DevKit)** — main MCU, Wi‑Fi, PSRAM for buffering and TinyML
* **ESP32‑CAM (OV2640)** — camera module
* **1 W white LED** (visible; for reflectance/scattering testing)
* **Logic MOSFET** for LED PWM switching
* **16 GB microSD card** for local logging
* **Breadboard + jumper wires** for prototyping
* **Li‑ion batteries** + buck converter (12 V → 5 V / 3.3 V)
* **Soldering iron & glue gun** and basic tools
* *(Planned / later)* **AS7265x** — multispectral sensor (this build targets AS7265x specifically)

**Calibration & Data‑collection notes**

* **Calibration flow:** dark frame → white (PTFE) reference → sample capture. Save all three files with timestamps and metadata (location, sample volume, illumination, exposure).
* **Preprocessing:** baseline subtraction (dark), reference division (normalize), low‑pass smoothing, and per‑channel scaling.
* **Model inputs:** spectral vector (18 channels) + image crops (localized particles) → decision fusion (CNN + small CNN for images).
* **Evaluation:** Precision/Recall for "microplastic" vs "non‑plastic"; confusion matrix for polymer class; size estimation MSE; FTIR validation on subset.

**Safety & lab notes**

* **Batteries:** use BMS, fuse, and safe charging procedures.
* **LEDs:** bright LEDs can damage eyes. Use eye protection for high‑intensity sources.

**References / Datasets**

1. Stressor‑AOP : [https://cb.imsc.res.in/saopadditives/](https://cb.imsc.res.in/saopadditives/)
2. NOAA NCEI Marine DB : [https://pmc.ncbi.nlm.nih.gov/articles/PMC10589325/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10589325/)
3. Marine Debris Program : [https://marinedebris.noaa.gov/](https://marinedebris.noaa.gov/)

---

*Changes made in this version:*

* Replaced UV illumination with a 1 W white LED.
* Replaced "1D‑CNN" with "CNN".
* Removed Random Forest from modelling options.
* Removed Bluetooth. System uses Wi‑Fi only.
* Explicit camera hardware set to ESP32‑CAM (OV2640).
