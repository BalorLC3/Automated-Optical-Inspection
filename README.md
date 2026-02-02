yolo-platform/
│
├── services/
│   ├── vision-cpp/          # C++ camera + preprocessing (edge)
│   │   ├── src/
│   │   ├── include/
│   │   ├── config/
│   │   ├── CMakeLists.txt
│   │   ├── k8s.yaml
│   │   └── README.md
│   │
│   ├── inference-py/        # Python YOLO training & inference
│   │   ├── src/             # train.py, serve.py, eval.py
│   │   ├── models/
│   │   ├── requirements.txt
│   │   ├── k8s.yaml
│   │   └── README.md
│   │
│   └── backend-go/          # Go IIoT backend / API / MQTT
│       ├── cmd/
│       ├── internal/
│       ├── api/             # OpenAPI / gRPC
│       ├── mqtt/
│       ├── k8s.yaml
│       └── README.md
│
├── platform/                # Shared infrastructure (cluster-wide)
│   ├── namespaces.yaml
│   ├── ingress.yaml
│   ├── mqtt.yaml            # Mosquitto / EMQX
│   ├── monitoring.yaml      # Prometheus / Grafana
│   └── storage.yaml
│
├── env/                     # Environment overlays
│   ├── dev/
│   ├── staging/
│   └── prod/
│
├── scripts/
│   ├── build.sh
│   ├── deploy.sh
│   └── teardown.sh
│
├── docs/
│   ├── architecture.md
│   ├── dataflow.md
│   └── deployment.md
│
└── README.md


Kubernetes-based modular AOI platform combining C++ real-time vision, Python ML inference, and Go-based IIoT backend for manufacturing environments.