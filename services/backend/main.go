package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"os"
	"path/filepath"
	"time"
)

// Config
const (
	Port         = ":8080"
	UploadDir    = "./data/uploads"
	InferenceURL = "http://localhost:8000/predict" // "inference" is the docker service name, for testing use localhost
)

// Response Structures
type BBox struct {
	X1 float64 `json:"x1"`
	Y1 float64 `json:"y1"`
	X2 float64 `json:"x2"`
	Y2 float64 `json:"y2"`
}

type Detection struct {
	ClassName  string  `json:"class_name"`
	ClassID    int     `json:"class_id"`
	Confidence float64 `json:"confidence"`
	BBox       BBox    `json:"bbox"`
}

type InferenceResponse struct {
	Filename   string      `json:"filename"`
	Detections []Detection `json:"detections"`
}

type APIResponse struct {
	Success  bool               `json:"success"`
	Message  string             `json:"message"`
	ImageURL string             `json:"image_url"`
	QCStatus string             `json:"qc_status"` // NEW: "PASS" or "FAIL"
	Data     *InferenceResponse `json:"data,omitempty"`
}

func main() {
	// Ensure upload directory exists
	if err := os.MkdirAll(UploadDir, 0755); err != nil {
		log.Fatal("Could not create upload directory:", err)
	}

	go startCleanupTicker() // When testing I created a lot of copies of images so this goroutine clean them up

	// Setup Routes
	http.HandleFunc("/api/process", enableCORS(handleProcess))

	// Serve static files (images) so frontend can see them
	fs := http.FileServer(http.Dir("./data"))
	http.Handle("/data/", http.StripPrefix("/data/", fs))

	log.Printf("Backend starting on port %s...", Port)
	log.Fatal(http.ListenAndServe(Port, nil))
}

// Middleware to allow Frontend to talk to Backend
func enableCORS(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next(w, r)
	}
}

func handleProcess(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse Uploaded File
	// Limit upload size to 10MB
	r.ParseMultipartForm(10 << 20)

	file, header, err := r.FormFile("image")
	if err != nil {
		jsonError(w, "Invalid file upload", http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Save File locally (simulating the 'Upload' phase)
	filename := fmt.Sprintf("%d_%s", time.Now().Unix(), header.Filename)
	filepath := filepath.Join(UploadDir, filename)

	dst, err := os.Create(filepath)
	if err != nil {
		jsonError(w, "Failed to save file", http.StatusInternalServerError)
		return
	}
	defer dst.Close()

	// Read file content to buffer for saving AND sending to Python
	fileBytes, _ := io.ReadAll(file)
	dst.Write(fileBytes) // Save to disk

	// Call Python Inference Service
	// (In the future, you inject the Generator call here)
	inferenceResult, err := callPythonService(filename, fileBytes)
	if err != nil {
		log.Printf("Python Inference Error: %v", err)
		jsonError(w, "AI Service Unavailable", http.StatusBadGateway)
		return
	}

	qcStatus := "PASS"
	rejectionReason := ""

	for _, detection := range inferenceResult.Detections {
		if detection.Confidence > 0.50 {
			qcStatus = "FAIL"
			rejectionReason = fmt.Sprintf("Detected %s (%.1f%%)", detection.ClassName, detection.Confidence*100)

			// Log to console
			log.Printf("[ALERT] Quality Control Rejected: %s - Reason %s", filename, rejectionReason)
			break
		}
	}
	finalMessage := "Quality Control Paused"
	if qcStatus == "FAIL" {
		finalMessage = "Quality Control Failed: " + rejectionReason
	}
	// Return JSON response
	response := APIResponse{
		Success:  true,
		Message:  finalMessage,
		QCStatus: qcStatus, // in frontend we will color code this
		ImageURL: fmt.Sprintf("/data/uploads/%s", filename),
		Data:     inferenceResult,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// Helper to send image to Python
// Helper to send image to Python
func callPythonService(filename string, fileData []byte) (*InferenceResponse, error) {
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	// 1. Detect the Content-Type (e.g., "image/jpeg") from the file bytes
	contentType := http.DetectContentType(fileData)

	// 2. Create the file part manually to set the Content-Type header
	h := make(textproto.MIMEHeader)
	h.Set("Content-Disposition", fmt.Sprintf(`form-data; name="file"; filename="%s"`, filename))
	h.Set("Content-Type", contentType)

	part, err := writer.CreatePart(h)
	if err != nil {
		return nil, err
	}
	part.Write(fileData)
	writer.Close()

	// 3. Send Request
	req, err := http.NewRequest("POST", InferenceURL, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		// Read body to see why it failed
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("python service status: %s | body: %s", resp.Status, string(bodyBytes))
	}

	var result InferenceResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return &result, nil
}

func startCleanupTicker() {
	ticker := time.NewTicker(10 * time.Minute) // Run every 10 minutes
	defer ticker.Stop()

	for range ticker.C {
		log.Println("Running cleanup task...")
		files, err := os.ReadDir(UploadDir)
		if err != nil {
			log.Printf("Error reading upload dir: %v", err)
			continue
		}

		cutoff := time.Now().Add(-15 * time.Minute) // Delete files older than 15 mins

		for _, file := range files {
			info, err := file.Info()
			if err != nil {
				continue
			}

			if info.ModTime().Before(cutoff) {
				fullPath := filepath.Join(UploadDir, file.Name())
				err := os.Remove(fullPath)
				if err != nil {
					log.Printf("Failed to delete %s: %v", file.Name(), err)
				} else {
					log.Printf("Deleted old file: %s", file.Name())
				}
			}
		}
	}
}

func jsonError(w http.ResponseWriter, message string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(APIResponse{
		Success: false,
		Message: message,
	})
}
