[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_black.png" width="150">](https://account.qbraid.com?gitHubUrl=<https://github.com/arashsm79/synqronix.git>)

# To run the code

1.  Click the "Launch on qBraid" button above. This will open a new qBraid Lab instance and clone this repository.
2.  Download the data from this link and put it in `/data` folder: https://gcell.umd.edu/. 
3.  Once qBraid Lab loads, run the `run_models.ipynb` notebook.
    (Adjust the parameters as needed.)

# 🧠 Mouse Auditory Cortex: Network Analysis with QGNNs  

This project is based on the study by Bowen et al., _"Fractured columnar small-world functional network organization in volumes of L2/3 of mouse auditory cortex,"_ [PNAS Nexus, 2024](https://doi.org/10.1093/pnasnexus/pgae074).
This is part of the **NeuroQuantum Nexus– Global Industry Challenge 2025**.

---

## 📚 Table of Contents
- 🚀 [Quick Start](#quick-start)
- 🧬 [Paper Summary](#paper-summary)
- 🗂️ [Data Structure](#data-structure)
- 🧾 [Data Format](#data-format)

---

## 🚀 Quick Start <a name="quick-start"></a>

1. Click **"Launch on qBraid"** above ☁️
2. Run `run_models.ipynb` in qBraid Lab 🧪
3. (Optional) Adjust parameters as needed 🛠️

---

## 🧬 Paper Summary <a name="paper-summary"></a>

- The auditory cortex (A1) is organized tonotopically—frequency preferences vary smoothly across space.
- In layer 2/3 (L2/3), neurons display **diverse tuning** but are organized in **microcolumns** by depth.
- Functional networks were built using in vivo two-photon imaging of ~1,000 neurons during sound stimulation.
- These networks show:
  - **Small-world properties**
  - **Rentian scaling**
  - **Spatial clustering of functionally similar neurons**
- Chronic 2P imaging of awake mice during tone presentation.
- Motion correction (Fourier registration)
- ROI detection (Chen et al. method)
- Neuropil subtraction: `F = Fsoma − 0.7 × FNP`
- Neurons identified as "responsive" via ANOVA (p < 0.01)
- Tuning curves generated from ΔF/F₀
- Best frequency (BF) = frequency with max average ΔF/F₀
- "Tuned" neurons = top 30% by tuning curve z-score

Suggesting L2/3 is organized for **efficient information processing** and **plasticity**.

---

## 🗂️ Data Structure <a name="data-structure"></a>

Directory tree:
```
data/
└── Auditory_cortex_data/
    ├── [session folders]/
    │   └── allPlanesVariables27-Feb-2021.mat
    ├── ant.m
    └── README.txt
```

Each `.mat` file contains info from ~370×370×100 µm cortical volumes with 100s of neurons.

---

## 🧾 Data Format <a name="data-format"></a>

This is the output of `process_mat()` on an example session `allPlanesVariables27-Feb-2021.mat`. This funciton converts the matlab file into a nice python dictionary.
Each plane is identified from 0 to 6. Here only plane 1 is shown as an example for some of the fields.
The shape or an example value or the value type of each key is shown.
```
BFInfo
    1
        cellnum: (8,)
        NONcellnum: (49,)
        BFval: (57,) # frequency id between 1-9 for each neuron
        BFresp: (57,)
        CFval: (57,)
        CFresp: (57,)
        BL: (57,)
        fraVals: (57, 9)
        bandwidth: (57,)
        normFRA: (57, 9)
        RFS: (57, 9)
        RFSBinary: (57, 9)
        sigRespCells: (8,)
        sigOffRespCells: (13,)
    ...
CellInfo
    1
        cellDists: (57, 57)
        cellAngles: (57, 57)
        sigTrial: (57, 9, 10)
        sigStim: (57, 9)
        sigOff: (57, 9)
    ...
CorrInfo
    1
        SigCorrs: (57, 57)
        NoiseCorrsTrial: (57, 57, 10)
        NoiseCorrsVec: (57, 57)
    ...
allZCorrInfo
    SigCorrs: (630, 630) # For layers with index 1 to 6
    NoiseCorrsTrial: (630, 630, 10)
allxc
    0: (1, 0)
    1: (57, 1)
    2: (114, 1)
    3: (121, 1)
    4: (124, 1)
    5: (114, 1)
    6: (100, 1)
allyc
    0: (1, 0)
    1: (57, 1)
    2: (114, 1)
    3: (121, 1)
    4: (124, 1)
    5: (114, 1)
    6: (100, 1)
allzc
    0: (1, 0)
    1: (57, 1)
    2: (114, 1)
    3: (121, 1)
    4: (124, 1)
    5: (114, 1)
    6: (100, 1)
zDFF
    0: (1, 0)
    1: (57, 2060)
    2: (114, 2060)
    3: (121, 2060)
    4: (124, 2060)
    5: (114, 2060)
    6: (100, 2060)
exptVars
    dimX: ()
    dimY: ()
    numImages: ()
    micronsPerPixel: ()
    frameRate: array(30, dtype=uint8)
    flybackFrames: ()
    stepSizeUM: ()
    numZSteps: ()
    totalZplanes: ()
    numVolumes: ()
    segmentSize: ()
selectZCorrInfo # only for layers with index 1 to 5
    SigCorrs: (530, 530) 
    NoiseCorrsTrial: (530, 530, 10)
stimInfo
    1
        Psignalfile: ()
        pfs: uint8
        PrimaryDuration: uint8
        PreStimSilence: uint8
        PostStimSilence: uint8
        BackgroundNoise: (3,)
        OverallDB: uint8
        Class: array('Tone      ', dtype='<U10')
        Trialindicies: (90, 3) : array([[8, 1, 0],
                                        [6, 1, 0],
                                        [6, 1, 0],
                                        ...
                                        [1, 1, 0],
                                        [7, 1, 0],
                                        [9, 1, 0]], dtype=uint8)
        framespertrial: array(90, dtype=uint8)
        FreqLevelOrder: (90,)
        Freqs: (90,)
        Levels: (90,)
        uFreqs: (9,)
        uLevels: array(70, dtype=uint8)
        sessionStartTime: MatlabOpaque((b'', b'MCOS', b'datetime', array([[3707764736],
                            [         2],
                            [         1],
                            [         1],
                            [         1],
                            [         1]], dtype=uint32)),
                                    dtype=[('s0', 'O'), ('s1', 'O'), ('s2', 'O'), ('arr', 'O')])
        soundTimes: MatlabOpaque((b'', b'MCOS', b'datetime', array([[3707764736],
                            [         2],
                            [         1],
                            [         1],
                            [         2],
                            [         1]], dtype=uint32)),
                                    dtype=[('s0', 'O'), ('s1', 'O'), ('s2', 'O'), ('arr', 'O')])
    ...
zStuff
    1
        flatFRA: (57, 9)
        flatTrialFRA: (57, 9, 10)
        pStim: (57, 9)
        pTrial: (57, 9, 10)
        zStimFrame: (90,)
        trialFreq: (9, 10, 23) # For each trial shows where there is silence, indicated by 0, and where there is stimulation, indicated by a non-zero value which is the actual value of the frequency.
        exptFreq: (2060,)
        trialDFF: (9, 10, 57, 23) # 9 frequencies 10 repetitions per frequency 57 neurons 23 time frames per trial. The frequency is an index into stimInfo/uFreqs to get the actual frequency.
    ...
```

Each neuron is stored in a structured dictionary or DataFrame with fields like:

| Field                | Description                                      |
|---------------------|--------------------------------------------------|
| `global_idx`         | Unique neuron ID (0–529)                        |
| `layer`              | Cortical layer (1–6)                            |
| `x, y, z`            | 3D coordinates                                  |
| `activity`           | Full-session trace                              |
| `per_trial_activity` | Shape: (9 freqs, 10 reps, T)                    |
| `BFval`, `BFresp`    | Best frequency + response                       |
| `PC`                 | First 4 PCA components                          |
| `global_corr`        | Correlation with all other neurons              |

---

