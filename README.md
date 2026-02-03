# Vehicle Position Prediction Using Deep Learning

A comprehensive research project for vehicle localization and position prediction using deep learning models combined with V2X communication-based data fusion algorithms.

## Table of Contents

1. [Overview](#overview)
2. [Research Background](#research-background)
3. [Key Features](#key-features)
4. [Technical Approach](#technical-approach)
5. [Repository Structure](#repository-structure)
6. [Dataset](#dataset)
7. [Deep Learning Models](#deep-learning-models)
8. [Data Fusion Algorithms](#data-fusion-algorithms)
9. [Installation](#installation)
10. [Usage](#usage)
11. [Experimental Results](#experimental-results)
12. [References](#references)

## Overview

This project implements a hybrid approach for precise vehicle position prediction by combining:
- **Deep Learning Models**: LSTM, GRU, and Transformer networks for time-series trajectory prediction
- **V2X Communication**: Vehicle-to-Everything communication for cooperative localization
- **Data Fusion**: Advanced algorithms to merge sensor measurements with deep learning predictions

The system achieves high positioning accuracy by leveraging both historical trajectory patterns learned by neural networks and real-time sensor measurements from connected vehicles.

## Research Background

### Motivation

Traditional vehicle positioning methods rely on:
- Single-vehicle sensors (GPS, cameras, LiDAR)
- Limited accuracy in complex environments
- Susceptibility to sensor errors and environmental interference

### V2X Communication & Cooperative Localization

V2X (Vehicle-to-Everything) communication enables:
- **V2V**: Vehicle-to-Vehicle communication
- **V2I**: Vehicle-to-Infrastructure communication  
- **V2P**: Vehicle-to-Pedestrian communication
- **V2N**: Vehicle-to-Network communication

Benefits:
- Real-time information sharing between vehicles
- Improved situational awareness
- Enhanced road safety
- Better traffic efficiency
- Collaborative decision-making

### Deep Learning for Position Prediction

Deep learning models can:
- Learn complex temporal patterns from vehicle trajectories
- Predict future positions based on historical data
- Handle non-linear relationships in vehicle dynamics
- Adapt to different driving scenarios

## Key Features

✅ Multiple deep learning architectures (LSTM, GRU, Transformer)  
✅ Stacking ensemble method for improved accuracy  
✅ Two novel data fusion algorithms  
✅ Real-world NGSIM dataset with noise filtering  
✅ Comprehensive evaluation metrics  
✅ Support for various vehicle network configurations (2-5 vehicles)

## Technical Approach

### Workflow

```
Raw Trajectory Data
        ↓
Data Smoothing (Savitzky-Golay Filter)
        ↓
Deep Learning Model Training
    ├── LSTM
    ├── GRU  
    ├── Transformer
    └── Stacking Ensemble
        ↓
Position Prediction
        ↓
Data Fusion with V2X Sensor Data
    ├── Observation Iteration Method
    └── Regional Optimization Method
        ↓
Refined Position Estimates
```

### Input Features

- Instantaneous velocity
- Instantaneous acceleration  
- Global X coordinate
- Global Y coordinate
- Time slot number

### Output Features

- Global X coordinate (predicted)
- Global Y coordinate (predicted)

## Repository Structure

```
Prediction_DL_Vehicle/
│
├── pre_slides.pptx                 # Research presentation (project overview)
│
├── datasmothing/                   # Data preprocessing
│   ├── sg_filter.py               # Savitzky-Golay filter implementation
│   └── NGSIM-US-101-trajectory-dataset-smoothing-master/
│       └── README.md              # Dataset smoothing documentation
│
├── GRU/                           # GRU model implementation
│   ├── smallsize_prediction.ipynb # GRU prediction notebook
│   ├── filtered.csv              # Processed dataset
│   ├── mememe.h5                 # Trained GRU model
│   ├── distances.txt             # Distance measurements
│   ├── trainingtime.png          # Training performance visualization
│   └── 旧的尝试/                  # Previous attempts
│       ├── train_GRU.py
│       ├── importData_GRU.py
│       └── GRU_predict.py
│
├── LSTM_2ndtrail/                 # LSTM model implementation (2nd trial)
│   ├── Model.ipynb               # LSTM model training
│   ├── data_processing.ipynb     # Data preprocessing
│   ├── DataSet.csv              # Dataset
│   ├── filtered.csv             # Processed data
│   └── model.h5                 # Trained LSTM model
│
├── GRU_2ndtrail/                  # GRU model implementation (2nd trial)
│   ├── Model.ipynb               # GRU model training
│   ├── data_processing.ipynb     # Data preprocessing
│   ├── DataSet.csv              # Dataset
│   └── model.h5                 # Trained GRU model
│
├── stacking/                      # Ensemble stacking method
│   ├── abstract.py               # Stacking algorithm pseudocode
│   ├── allnew50.ipynb           # 50-epoch training
│   ├── allnew200.ipynb          # 200-epoch training
│   ├── smallsize_predictionGRU.ipynb
│   ├── smallsize_predictionLSTM.ipynb
│   ├── gru.h5                   # GRU component model
│   ├── LSTM.h5                  # LSTM component model
│   ├── filtered.csv             # Processed data
│   └── distances.txt            # Distance measurements
│
├── 3stepPre/                      # 3-step prediction approach
│   └── GRU/
│       ├── 3stepPre.h5
│       ├── smallsize_prediction.ipynb
│       └── filtered.csv
│
└── 2nd_step/                      # Data fusion algorithms
    ├── mainIdea.pdf              # Core algorithm documentation
    ├── mainIdea.docx
    ├── relocalization_pre.ipynb  # Preliminary relocalization
    ├── relocalization4.ipynb     # 4-vehicle relocalization
    ├── relocalization5.ipynb     # 5-vehicle relocalization
    ├── new_streamlined_method1.ipynb
    ├── draw.py                   # Visualization utilities
    ├── mememe.h5                 # Model file
    ├── *.csv                     # Various test datasets (1551, 1564, 1583, 1629, 1638)
    ├── *.png                     # Result visualizations
    ├── *.docx                    # Documentation
    ├── 5车定位.txt               # 5-vehicle positioning results
    ├── 1method/                  # Method 1 results and visualizations
    │   └── *.png                 # Various performance plots
    └── oldway/                   # Legacy implementations
        ├── 2nddataprocess.py
        ├── 3nd.py
        └── 4th.py
```

## Dataset

### NGSIM Dataset

The project uses the **Next Generation Simulation (NGSIM)** dataset:

- **Source**: US Highway 101 (Hollywood Freeway, Los Angeles)
- **Collection Method**: 8 cameras mounted on buildings
- **Data Type**: Vehicle trajectories with position, velocity, acceleration
- **Time Period**: Real-world traffic data

### Data Preprocessing

1. **Noise Filtering**: Savitzky-Golay filter
   - Removes measurement noise while preserving signal shape
   - Applied to Local X and Y coordinates
   - Recomputes velocities and accelerations from smoothed positions

2. **Known Issues**:
   - Original dataset contains unrealistic accelerations (>3 m/s²)
   - 8.99% of data shows physically impossible vehicle dynamics
   - No official accuracy assessment from NGSIM documentation

### Training Configuration

- **Input Window**: 3600 seconds (60 minutes) of historical data
- **Training Epochs**: 50 epochs for main models
- **Dataset Split**: Peachtree Street dataset for data fusion validation

## Deep Learning Models

### 1. Long Short-Term Memory (LSTM)

```
Input: (batch_size, sequence_length, features)
   ↓
LSTM Layers (capture long-term dependencies)
   ↓
Dense Layer
   ↓
Output: (x, y) coordinates
```

**Characteristics**:
- Handles long-term dependencies through memory cells
- Three gates: input, forget, output
- Good for sequential trajectory prediction

### 2. Gated Recurrent Unit (GRU)

```
Input: (batch_size, 10, 5)  # 10 time steps, 5 features
   ↓
GRU Layer (64 hidden units)
   ↓
Lambda Layer (take last time step)
   ↓
Dense Layer (2 outputs: x, y)
```

**Configuration**:
- Hidden size: 64 units
- Optimizer: Adam
- Loss function: Mean Squared Error (MSE)
- Batch size: 8
- Epochs: 200

**Advantages**:
- Simpler architecture than LSTM (fewer parameters)
- Faster training
- Better fitting results in experiments

### 3. Transformer

**Architecture**:
- 6 Encoder layers
- 6 Decoder layers
- Multi-head attention mechanism
- Positional encoding
- Residual connections

**Features**:
- Parallel processing of sequences
- Attention mechanism for capturing dependencies
- State-of-the-art for sequence modeling

### 4. Stacking Ensemble Method

Combines multiple models for improved performance:

```python
# Pseudocode from stacking/abstract.py
X_train, y_train = load_data()
X_test, y_test = load_data()

# Train base models
gru_pred_train = GRU_model.predict(X_train)
gru_pred_test = GRU_model.predict(X_test)

lstm_pred_train = LSTM_model.predict(X_train)
lstm_pred_test = LSTM_model.predict(X_test)

# Stack predictions as features
X_train_stacked = [X_train, gru_pred_train, lstm_pred_train]
X_test_stacked = [X_test, gru_pred_test, lstm_pred_test]

# Meta-learner: Random Forest
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train_stacked, y_train)
y_pred = rf.predict(X_test_stacked)
```

**Benefits**:
- Leverages strengths of different models
- Reduces model variance
- Highest prediction accuracy in experiments

## Data Fusion Algorithms

After obtaining position predictions from deep learning models, data fusion algorithms refine predictions using V2X sensor data (inter-vehicle distances and angles).

### Core Concept

**Sensor measurements + Deep learning predictions → Refined positions**

### Method 1: Observation Iteration Method

**Suitable for**: Networks with ≥5 connected vehicles

**Approach**:
1. Iteratively observe each vehicle (vehicle A) 
2. For each observation, use the nearest vehicle (vehicle B)
3. Calculate observation-based position estimate
4. Evaluate using error metric M:
   - M = |sensor_measured_distance - observed_distance|
5. Update position if M improves

**Process**:
```
For vehicle A:
  For each nearby vehicle B:
    - Get sensor measurement: distance L_AB, angle θ
    - Compute observed position based on B's position
    - Calculate error M
    - If M decreases, update position
  Next vehicle
Next vehicle A
```

**Performance**:
- **5 vehicles**: All vehicles improved
- **4 vehicles**: 81 out of 100 vehicles improved  
- **3 vehicles**: 33 out of 75 vehicles improved
- **2 vehicles**: 17 out of 50 vehicles improved

### Method 2: Regional Optimization Method

**Suitable for**: Networks with exactly 4 connected vehicles

**Approach**: Function optimization using L-BFGS algorithm

1. **Formulation**: Define error function for vehicle A:
   ```
   F(A) = Σ (distance_error_i)²
   where distance_error = |predicted_distance - sensor_measured_distance|
   ```

2. **Optimization**: Use L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)
   - Quasi-Newton method for large-scale optimization
   - Efficient for high-dimensional problems
   - Finds position that minimizes total distance error

3. **Constraints**: Bounded optimization within reasonable search space

**Performance**:
- All vehicles in 4-vehicle networks improved
- Consistent improvement across 25 test scenarios

**Reference**: Zhu et al., "Algorithm 778: L-BFGS-B: Fortran subroutines for large-scale bound-constrained optimization," ACM TOMS, 1997.

## Installation

### Requirements

```bash
# Python 3.7+
python --version

# Core dependencies
pip install numpy pandas matplotlib scipy
pip install tensorflow keras  # or pytorch
pip install scikit-learn
pip install jupyter notebook
pip install python-pptx  # for reading presentation files
```

### Optional

```bash
pip install seaborn  # for enhanced visualizations
pip install xlsxwriter pillow lxml  # for additional file format support
```

## Usage

### 1. Data Preprocessing

```bash
cd datasmothing
python sg_filter.py
```

### 2. Train Deep Learning Models

**GRU Model**:
```bash
cd GRU
jupyter notebook smallsize_prediction.ipynb
# or
python 旧的尝试/train_GRU.py
```

**LSTM Model**:
```bash
cd LSTM_2ndtrail
jupyter notebook Model.ipynb
```

**Stacking Ensemble**:
```bash
cd stacking
jupyter notebook allnew200.ipynb
```

### 3. Data Fusion

**Observation Iteration Method** (5 vehicles):
```bash
cd 2nd_step
jupyter notebook relocalization5.ipynb
```

**Regional Optimization Method** (4 vehicles):
```bash
cd 2nd_step
jupyter notebook relocalization4.ipynb
```

### 4. Visualization

```bash
cd 2nd_step
python draw.py
```

## Experimental Results

### Model Comparison

| Model | MSE | MAE | Mean Euclidean Distance | Training Time |
|-------|-----|-----|------------------------|---------------|
| LSTM | Higher | Higher | Higher | Longer |
| GRU | Lower | Lower | Lower | Moderate |
| Transformer | Moderate | Moderate | Moderate | Longer |
| **Stacking** | **Lowest** | **Lowest** | **Lowest** | Longest |

**Key Finding**: GRU outperforms LSTM, and the Stacking ensemble achieves the highest overall accuracy.

### Data Fusion Performance

#### Observation Iteration Method

- **5-vehicle network**: 100% of vehicles improved
- **4-vehicle network**: 81% of vehicles improved (25 test scenarios)
- **3-vehicle network**: 44% of vehicles improved  
- **2-vehicle network**: 34% of vehicles improved

**Conclusion**: Method works best with ≥5 connected vehicles

#### Regional Optimization Method

- **4-vehicle network**: 100% of vehicles improved across all 25 test scenarios
- Consistent performance
- Reliable for exactly 4-vehicle configurations

### Evaluation Metrics

1. **Mean Square Error (MSE)**: 
   - Measures average squared difference between predicted and actual positions
   - Lower is better

2. **Mean Absolute Error (MAE)**:
   - Average absolute difference in position
   - More interpretable than MSE

3. **Mean Euclidean Distance**:
   - Average straight-line distance between predicted and actual positions
   - Direct measure of positioning accuracy

4. **Improvement Rate**:
   - Percentage of vehicles with reduced positioning error after data fusion

## Project Highlights

### Innovation

1. **Hybrid Approach**: Combines deep learning with V2X sensor fusion
2. **Multiple Algorithms**: Two data fusion methods for different network sizes
3. **Ensemble Learning**: Stacking method leverages multiple model strengths
4. **Real-World Data**: Validated on NGSIM dataset with noise filtering

### Advantages

- ✅ Higher positioning accuracy than single-method approaches
- ✅ Robust to sensor noise through data fusion
- ✅ Scalable to different vehicle network configurations
- ✅ Real-time capable with pre-trained models

### Limitations & Future Work

1. **Network Requirements**: 
   - Requires V2X-enabled vehicles
   - Performance degrades with <4 vehicles

2. **Computational Complexity**:
   - Deep learning models need GPU for training
   - Real-time inference requires optimization

3. **Dataset Limitations**:
   - NGSIM data contains inherent noise
   - Limited to highway scenarios

4. **Future Directions**:
   - Extend to urban environments
   - Incorporate additional sensor modalities (LiDAR, cameras)
   - Develop lightweight models for edge deployment
   - Explore transformer-based fusion architectures

## References

### Key Research Papers

1. Liu et al., "Cloud-assisted cooperative localization for vehicle platoons: A turbo approach," IEEE Transactions on Signal Processing, 2020.

2. Rohani et al., "A new decentralized Bayesian approach for cooperative vehicle localization based on fusion of GPS and VANET," IEEE Intelligent Transportation Systems Magazine, 2015.

3. Kang et al., "Lidar-and V2X-Based Cooperative Localization Technique for Autonomous Driving in a GNSS-Denied Environment," Remote Sensing, 2022.

4. Song et al., "Blockchain-enabled internet of vehicles with cooperative positioning: A deep neural network approach," IEEE Internet of Things Journal, 2020.

5. Zhu et al., "Algorithm 778: L-BFGS-B: Fortran subroutines for large-scale bound-constrained optimization," ACM TOMS, 1997.

### Dataset References

6. Punzo et al., "On the assessment of vehicle trajectory data accuracy and application to the Next Generation SIMulation (NGSIM) program data," Transportation Research Part C, 2011.

7. Lu & Skabardonis, "Freeway traffic shockwave analysis: exploring the NGSIM trajectory data," 86th TRB Annual Meeting, 2007.

8. Montanino & Punzo, "Making NGSIM data usable for studies on traffic flow theory," Transportation Research Record, 2013.

9. Thiemann et al., "Estimating acceleration and lane-changing dynamics from next generation simulation trajectory data," Transportation Research Record, 2008.

10. Altché & de La Fortelle, "An LSTM network for highway trajectory prediction," IEEE ITSC, 2017.

### Official Resources

- [NGSIM Dataset](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj)
- [NGSIM Fact Sheet](https://www.fhwa.dot.gov/publications/research/operations/07030/index.cfm)
- [Savitzky-Golay Filter Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html)

## Contributing

This is a research project. For questions or collaborations, please refer to the presentation file (`pre_slides.pptx`) for detailed methodology and results.

## License

Please check with the repository owner for licensing information.

## Acknowledgments

- NGSIM dataset providers
- Open-source deep learning frameworks (TensorFlow/Keras, PyTorch)
- Scipy for signal processing tools
- Research community for V2X communication and cooperative localization

---

**Note**: This project demonstrates the potential of combining deep learning with V2X communication for enhanced vehicle positioning. Results shown are from simulation and require further validation in real-world deployments.
