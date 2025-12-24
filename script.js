// Facial Recognition and Database Management System
// Author: woshitaiguorennine-alt
// Created: 2025-12-24

const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const face = require('@vladmandic/face-api');
const canvas = require('canvas');
const fs = require('fs');
const path = require('path');

// Initialize Express App
const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Setup Canvas for Face-API
const { Canvas, Image, ImageData } = canvas;
face.env.monkeyPatch({ Canvas, Image, ImageData });

// Initialize SQLite Database
const db = new sqlite3.Database('./faces.db', (err) => {
  if (err) {
    console.error('Database connection error:', err);
  } else {
    console.log('Connected to SQLite database');
    initializeDatabase();
  }
});

// Initialize Database Schema
function initializeDatabase() {
  const createTableSQL = `
    CREATE TABLE IF NOT EXISTS faces (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      descriptor TEXT NOT NULL,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
      image_path TEXT
    );
    
    CREATE TABLE IF NOT EXISTS recognition_logs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      matched_name TEXT,
      confidence REAL,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
      status TEXT
    );
  `;

  db.exec(createTableSQL, (err) => {
    if (err) {
      console.error('Error creating tables:', err);
    } else {
      console.log('Database tables initialized successfully');
    }
  });
}

// Load Face Detection Models
async function loadModels() {
  try {
    const modelPath = path.join(__dirname, 'models');
    await face.nets.tinyFaceDetector.load(modelPath);
    await face.nets.faceLandmark68Net.load(modelPath);
    await face.nets.faceRecognitionNet.load(modelPath);
    console.log('Face detection models loaded successfully');
  } catch (error) {
    console.error('Error loading models:', error);
  }
}

// Detect Face and Extract Descriptors
async function detectFace(imagePath) {
  try {
    const img = await canvas.loadImage(imagePath);
    const detections = await face.detectAllFaces(img).withFaceLandmarks().withFaceDescriptors();
    return detections;
  } catch (error) {
    console.error('Error detecting face:', error);
    return null;
  }
}

// Register New Face
function registerFace(name, descriptor, imagePath) {
  return new Promise((resolve, reject) => {
    const descriptorJSON = JSON.stringify(descriptor);
    const sql = `INSERT INTO faces (name, descriptor, image_path) VALUES (?, ?, ?)`;
    
    db.run(sql, [name, descriptorJSON, imagePath], function(err) {
      if (err) {
        reject(err);
      } else {
        console.log(`Face registered for ${name} with ID: ${this.lastID}`);
        resolve(this.lastID);
      }
    });
  });
}

// Retrieve All Registered Faces
function getAllFaces() {
  return new Promise((resolve, reject) => {
    const sql = `SELECT id, name, descriptor FROM faces`;
    
    db.all(sql, [], (err, rows) => {
      if (err) {
        reject(err);
      } else {
        const faces = rows.map(row => ({
          id: row.id,
          name: row.name,
          descriptor: JSON.parse(row.descriptor)
        }));
        resolve(faces);
      }
    });
  });
}

// Calculate Euclidean Distance Between Descriptors
function calculateDistance(descriptor1, descriptor2) {
  let sum = 0;
  for (let i = 0; i < descriptor1.length; i++) {
    const diff = descriptor1[i] - descriptor2[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

// Recognize Face from Image
async function recognizeFace(imagePath, threshold = 0.5) {
  try {
    const detections = await detectFace(imagePath);
    
    if (!detections || detections.length === 0) {
      return { success: false, message: 'No face detected in image' };
    }

    const registeredFaces = await getAllFaces();
    const results = [];

    for (const detection of detections) {
      const inputDescriptor = detection.descriptor;
      let bestMatch = { name: 'Unknown', distance: Infinity };

      for (const registeredFace of registeredFaces) {
        const distance = calculateDistance(inputDescriptor, registeredFace.descriptor);
        
        if (distance < bestMatch.distance) {
          bestMatch = {
            name: registeredFace.name,
            distance: distance
          };
        }
      }

      const isMatch = bestMatch.distance < threshold;
      const confidence = Math.max(0, 100 - (bestMatch.distance * 100));

      results.push({
        name: bestMatch.name,
        confidence: confidence.toFixed(2),
        isMatched: isMatch,
        distance: bestMatch.distance
      });

      // Log recognition attempt
      logRecognition(bestMatch.name, confidence, isMatch);
    }

    return { success: true, results: results };
  } catch (error) {
    console.error('Error recognizing face:', error);
    return { success: false, message: 'Error during face recognition', error: error.message };
  }
}

// Log Recognition Attempt
function logRecognition(matchedName, confidence, status) {
  return new Promise((resolve, reject) => {
    const sql = `INSERT INTO recognition_logs (matched_name, confidence, status) VALUES (?, ?, ?)`;
    const statusText = status ? 'matched' : 'unmatched';
    
    db.run(sql, [matchedName, confidence, statusText], function(err) {
      if (err) {
        reject(err);
      } else {
        resolve(this.lastID);
      }
    });
  });
}

// Get Recognition History
function getRecognitionHistory(limit = 50) {
  return new Promise((resolve, reject) => {
    const sql = `SELECT * FROM recognition_logs ORDER BY timestamp DESC LIMIT ?`;
    
    db.all(sql, [limit], (err, rows) => {
      if (err) {
        reject(err);
      } else {
        resolve(rows);
      }
    });
  });
}

// Delete Face from Database
function deleteFace(faceId) {
  return new Promise((resolve, reject) => {
    const sql = `DELETE FROM faces WHERE id = ?`;
    
    db.run(sql, [faceId], function(err) {
      if (err) {
        reject(err);
      } else {
        console.log(`Face with ID ${faceId} deleted`);
        resolve(true);
      }
    });
  });
}

// API Endpoints

// Register a new face
app.post('/api/register', async (req, res) => {
  try {
    const { name, imagePath } = req.body;
    
    if (!name || !imagePath) {
      return res.status(400).json({ error: 'Name and image path are required' });
    }

    const detections = await detectFace(imagePath);
    
    if (!detections || detections.length === 0) {
      return res.status(400).json({ error: 'No face detected in the provided image' });
    }

    if (detections.length > 1) {
      return res.status(400).json({ error: 'Multiple faces detected. Please provide an image with only one face' });
    }

    const descriptor = detections[0].descriptor;
    const faceId = await registerFace(name, descriptor, imagePath);

    res.json({
      success: true,
      message: `Face registered successfully for ${name}`,
      faceId: faceId
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Recognize a face
app.post('/api/recognize', async (req, res) => {
  try {
    const { imagePath, threshold } = req.body;
    
    if (!imagePath) {
      return res.status(400).json({ error: 'Image path is required' });
    }

    const result = await recognizeFace(imagePath, threshold || 0.5);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get all registered faces
app.get('/api/faces', async (req, res) => {
  try {
    const faces = await getAllFaces();
    res.json({
      success: true,
      count: faces.length,
      faces: faces.map(f => ({ id: f.id, name: f.name }))
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get recognition history
app.get('/api/history', async (req, res) => {
  try {
    const limit = req.query.limit || 50;
    const history = await getRecognitionHistory(limit);
    res.json({
      success: true,
      count: history.length,
      history: history
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Delete a face
app.delete('/api/faces/:id', async (req, res) => {
  try {
    const faceId = req.params.id;
    await deleteFace(faceId);
    res.json({
      success: true,
      message: `Face with ID ${faceId} deleted successfully`
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

// Initialize and Start Server
async function startServer() {
  try {
    await loadModels();
    const PORT = process.env.PORT || 3000;
    app.listen(PORT, () => {
      console.log(`Facial Recognition Server running on port ${PORT}`);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
  }
}

// Graceful Shutdown
process.on('SIGINT', () => {
  db.close((err) => {
    if (err) {
      console.error('Error closing database:', err);
    } else {
      console.log('Database connection closed');
    }
    process.exit(0);
  });
});

// Export functions for external use
module.exports = {
  detectFace,
  registerFace,
  recognizeFace,
  getAllFaces,
  getRecognitionHistory,
  deleteFace,
  app
};

// Start the server if this is the main module
if (require.main === module) {
  startServer();
}
