# Smart Attendance Monitoring System

An AI-powered attendance monitoring system using face detection, face recognition, dual-camera monitoring, and MySQL-based attendance management.

## 🚀 Overview

The Smart Attendance Monitoring System automates employee/student attendance using computer vision.

The system uses:

- **YuNet** for face detection
- **SFace** for face recognition
- **Camera 1** for Entry / Login
- **Camera 2** for Exit / Logout
- **MySQL** for attendance storage
- **Python + OpenCV** for real-time processing

When a registered person is detected at the entry camera, the system creates an **ACTIVE** attendance session.

When the same person is detected at the exit camera, the active session is closed and marked **COMPLETED**.

Unknown faces are detected but are not recorded in the attendance database.

---

## ✨ Features

- Real-time face detection
- Face recognition using SFace
- YuNet deep-learning face detector
- Multiple faces detected simultaneously
- Dual-camera entry and exit monitoring
- Automatic login and logout
- MySQL attendance database
- Multiple attendance sessions per day
- Unknown faces are ignored
- IP camera support
- Laptop/webcam support
- Environment-variable based configuration
- Parameterized SQL queries
- Biometric data excluded from GitHub
- Sensitive configuration excluded using `.gitignore`

---

## 🏗️ System Architecture

```text
                 ┌──────────────────────┐
                 │      Entry Camera    │
                 │      Camera 1        │
                 └──────────┬───────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │      YuNet      │
                   │ Face Detection  │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │      SFace      │
                   │ Face Recognition│
                   └────────┬────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Identity Match│
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  LOGIN / DB   │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │     MySQL     │
                    │   Attendance  │
                    └───────────────┘


                 ┌──────────────────────┐
                 │       Exit Camera    │
                 │       Camera 2       │
                 └──────────┬───────────┘
                            │
                            ▼
                   YuNet + SFace
                            │
                            ▼
                     Identity Match
                            │
                            ▼
                    LOGOUT / UPDATE
                            │
                            ▼
                         MySQL