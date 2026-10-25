# 🏛️ System Architecture Specification

## 1. Executive Summary
The **Automated Fruit Ripeness Detection & Classification System** is an enterprise-grade Computer Vision application. The system eliminates subjectivity and manual delays in agricultural sorting, grocery inventory management, and post-harvest quality assessment by performing non-destructive optical ripeness grading across **131 distinct fruit categories and varieties**.

---

## 2. Three-Tier Architectural Topology

The application adheres to a decoupled **Three-Tier Architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│                 PRESENTATION TIER (CLIENT)                  │
│                                                             │
│  ┌───────────────────────────┐ ┌──────────────────────────┐ │
│  │ Streamlit Web Dashboard   │ │ Tkinter Desktop App      │ │
│  │ (Interactive Analytics)   │ │ (Offline GUI Prototype)  │ │
│  └─────────────┬─────────────┘ └────────────┬─────────────┘ │
└────────────────┼────────────────────────────┼───────────────┘
                 │ HTTP / Local Interface     │
                 ▼                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 APPLICATION / LOGIC TIER                    │
│                                                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Deep Learning Inference Engine (TensorFlow / Keras)    │ │
│  │ 2-Stage Sequential CNN (Conv2D -> MaxPool -> Dropout)  │ │
│  └───────────────────────────┬────────────────────────────┘ │
│                              │                              │
│  ┌───────────────────────────┴────────────────────────────┐ │
│  │ Heuristic Fallback & Colorimetric Feature Extraction   │ │
│  │ (Dominant RGB, HSV, Saturation & Texture Indices)      │ │
│  └───────────────────────────┬────────────────────────────┘ │
│                              │                              │
│  ┌───────────────────────────┴────────────────────────────┐ │
│  │ User Authentication & Audit Controller                 │ │
│  └───────────────────────────┬────────────────────────────┘ │
└──────────────────────────────┼──────────────────────────────┘
                               │ SQL Query Interface
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     DATA TIER (STORAGE)                     │
│                                                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ SQLite Relational Database Engine (`fruit_app.db`)     │ │
│  │ • Table: `users` (Authentication & Role Management)    │ │
│  │ • Table: `images` (Metadata & File Reference Paths)    │ │
│  │ • Table: `predictions` (Audit Logs & Quality Scores)   │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Deep Learning Pipeline & Model Specification

### 3.1 Preprocessing Pipeline
1. **Input Rescaling:** Rescaled from 8-bit RGB `[0, 255]` to normalized float32 `[0.0, 1.0]` (`rescale=1./255`).
2. **Standard Dimensionality:** Target image dimensions fixed at `64 x 64 x 3` channels.
3. **Data Augmentation:** Real-time geometric transformations:
   - Shear range: `0.2`
   - Zoom range: `0.2`
   - Horizontal flip: `True`

### 3.2 CNN Layer Architecture
| Layer Index | Layer Type | Parameters / Filters | Activation | Output Shape | Receptive Field Role |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | Input | 64 x 64 x 3 | — | `(None, 64, 64, 3)` | Raw RGB Tensor |
| **1** | Conv2D | 64 filters (3x3 kernel) | ReLU | `(None, 62, 62, 64)` | Low-level edge & color feature detection |
| **2** | MaxPool2D | Pool: 2x2, Strides: 2 | — | `(None, 31, 31, 64)` | Spatial downsampling (translation invariance) |
| **3** | Conv2D | 64 filters (3x3 kernel) | ReLU | `(None, 29, 29, 64)` | Higher-level texture & morphology mapping |
| **4** | MaxPool2D | Pool: 2x2, Strides: 2 | — | `(None, 14, 14, 64)` | Spatial feature reduction |
| **5** | Dropout | Rate: 0.50 | — | `(None, 14, 14, 64)` | Overfitting prevention & regularizer |
| **6** | Flatten | 12,544 nodes | — | `(None, 12544)` | Flattening spatial maps to 1D vector |
| **7** | Dense | 256 neurons | ReLU | `(None, 256)` | Deep non-linear semantic fusion |
| **8** | Dense (Output) | 131 units | Softmax | `(None, 131)` | Multi-class categorical probability distribution |

---

## 4. Database Schema Design

### `users` Table
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,
    role TEXT DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### `images` Table
```sql
CREATE TABLE images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    image_path TEXT NOT NULL,
    tags TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
);
```

### `predictions` Table
```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    fruit_name TEXT NOT NULL,
    ripeness TEXT NOT NULL,
    confidence REAL NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
);
```
