package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// Config
const (
	Port         = ":8080"
	UploadDir    = "./data/uploads"
	InferenceURL = "http://inference:8000/predict" // "inference" is the docker service name
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
	Data     *InferenceResponse `json:"data,omitempty"`
}

func main() {
	// 1. Ensure upload directory exists
	if err := os.MkdirAll(UploadDir, 0755); err != nil {
		log.Fatal("Could not create upload directory:", err)
	}

	// 2. Setup Routes
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

	// Return JSON response
	response := APIResponse{
		Success:  true,
		Message:  "Processed successfully",
		ImageURL: fmt.Sprintf("/data/uploads/%s", filename),
		Data:     inferenceResult,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// Helper to send image to Python
func callPythonService(filename string, fileData []byte) (*InferenceResponse, error) {
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	part, err := writer.CreateFormFile("file", filename)
	if err != nil {
		return nil, err
	}
	part.Write(fileData)
	writer.Close()

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
		return nil, fmt.Errorf("python service status: %s", resp.Status)
	}

	var result InferenceResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return &result, nil
}

func jsonError(w http.ResponseWriter, message string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(APIResponse{
		Success: false,
		Message: message,
	})
}
