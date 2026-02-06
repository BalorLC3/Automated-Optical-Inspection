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
	InferenceURL = "http://backend:8000/predict" // for development use localhost:
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
	Filename       string      `json:"filename"`
	Detections     []Detection `json:"detections"`
	ProcessedImage string      `json:"processed_image"` // base64 image with boxes
}

type APIResponse struct {
	Success  bool          `json:"success"`
	Message  string        `json:"message"`
	ImageURL string        `json:"image_url"`
	QCStatus string        `json:"qcStatus"` // Changed to match frontend
	Data     *ResponseData `json:"data,omitempty"`
}

type ResponseData struct {
	Detections     []FrontendDetection `json:"detections"`
	ProcessedImage string              `json:"processedImage"` // base64 image
}

type FrontendDetection struct {
	X          float64 `json:"x"`
	Y          float64 `json:"y"`
	W          float64 `json:"w"`
	H          float64 `json:"h"`
	Label      string  `json:"label"`
	Confidence float64 `json:"confidence"`
}

func main() {
	if err := os.MkdirAll(UploadDir, 0755); err != nil {
		log.Fatal("Could not create upload directory:", err)
	}

	go startCleanupTicker()

	http.HandleFunc("/api/process", enableCORS(handleProcess))
	fs := http.FileServer(http.Dir("./data"))
	http.Handle("/data/", http.StripPrefix("/data/", fs))

	log.Printf("Backend starting on port %s...", Port)
	log.Fatal(http.ListenAndServe(Port, nil))
}

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

	r.ParseMultipartForm(10 << 20)

	file, header, err := r.FormFile("image")
	if err != nil {
		jsonError(w, "Invalid file upload", http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Save file locally
	filename := fmt.Sprintf("%d_%s", time.Now().Unix(), header.Filename)
	filepath := filepath.Join(UploadDir, filename)

	dst, err := os.Create(filepath)
	if err != nil {
		jsonError(w, "Failed to save file", http.StatusInternalServerError)
		return
	}
	defer dst.Close()

	fileBytes, _ := io.ReadAll(file)
	dst.Write(fileBytes)

	// Call Python service
	inferenceResult, err := callPythonService(filename, fileBytes)
	if err != nil {
		log.Printf("Python Inference Error: %v", err)
		jsonError(w, "AI Service Unavailable", http.StatusBadGateway)
		return
	}

	// Determine QC status
	qcStatus := "PASS"
	rejectionReason := ""

	for _, detection := range inferenceResult.Detections {
		if detection.Confidence > 0.50 {
			qcStatus = "FAIL"
			rejectionReason = fmt.Sprintf("Detected %s (%.1f%%)", detection.ClassName, detection.Confidence*100)
			log.Printf("[ALERT] Quality Control Rejected: %s - Reason %s", filename, rejectionReason)
			break
		}
	}

	finalMessage := "No defects detected"
	if qcStatus == "FAIL" {
		finalMessage = fmt.Sprintf("Detected %d defect(s)", len(inferenceResult.Detections))
	}

	// Convert detections to frontend format
	frontendDetections := make([]FrontendDetection, len(inferenceResult.Detections))
	for i, det := range inferenceResult.Detections {
		frontendDetections[i] = FrontendDetection{
			X:          det.BBox.X1,
			Y:          det.BBox.Y1,
			W:          det.BBox.X2 - det.BBox.X1,
			H:          det.BBox.Y2 - det.BBox.Y1,
			Label:      det.ClassName,
			Confidence: det.Confidence,
		}
	}

	// Return response with processed image
	response := APIResponse{
		Success:  true,
		Message:  finalMessage,
		QCStatus: qcStatus,
		ImageURL: fmt.Sprintf("/data/uploads/%s", filename),
		Data: &ResponseData{
			Detections:     frontendDetections,
			ProcessedImage: inferenceResult.ProcessedImage, // pass through base64 image
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func callPythonService(filename string, fileData []byte) (*InferenceResponse, error) {
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	contentType := http.DetectContentType(fileData)

	h := make(textproto.MIMEHeader)
	h.Set("Content-Disposition", fmt.Sprintf(`form-data; name="file"; filename="%s"`, filename))
	h.Set("Content-Type", contentType)

	part, err := writer.CreatePart(h)
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
	ticker := time.NewTicker(10 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		log.Println("Running cleanup task...")
		files, err := os.ReadDir(UploadDir)
		if err != nil {
			log.Printf("Error reading upload dir: %v", err)
			continue
		}

		cutoff := time.Now().Add(-15 * time.Minute)

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
