
Athena Security Guardian

An Edge AI-Powered Virtual Security Assistant
Built for the Snapdragon Multiverse Hackathon @ Princeton.

⸻

🚀 Overview

Athena Security Guardian is an AI-powered edge application that redefines physical security for offices, hospitals, residential complexes, and public spaces.

Unlike traditional systems that rely on costly human monitoring or cloud-based AI (which introduce latency and privacy risks), Athena operates entirely on-device using Snapdragon-powered laptops and phones.

This ensures:
	•	⚡ Ultra-low latency decisions (<50ms)
	•	🔒 Privacy-first operations (no cloud dependency, no biometric data leaving the device)
	•	🛡️ Always-on reliability (works even offline)

Athena acts as both:
	1.	A virtual security guard – monitoring, detecting, and alerting in real-time.
	2.	An intelligent assistant – answering visitor queries and escalating when needed.

⸻

🧩 Key Features
	•	👁️ Edge-based Motion & Person Detection
	•	YOLOv8 (quantized ONNX/TFLite optimized for Snapdragon NPU)
	•	Runs at 30+ FPS with <50ms latency
	•	🧑‍🤝‍🧑 Facial Recognition & Role Classification
	•	MobileFaceNet embeddings for resident/staff vs. unknown visitor differentiation
	•	GDPR/CCPA compliant (no cloud storage of biometrics)
	•	🎙️ Voice Interaction
	•	On-device Whisper-small (quantized) for speech-to-text
	•	DistilBERT intent classifier for query understanding
	•	On-device TTS for natural responses
	•	🚨 Security Threat Detection
	•	Identifies tailgating, loitering, restricted entry attempts
	•	Triggers instant audio alerts + optional push notifications
	•	🛡️ Offline & Privacy-first
	•	Works without internet
	•	Session-only embeddings (no permanent storage)

⸻

🏗️ Technical Architecture

Hardware: Snapdragon-powered Laptop + Mobile Device
Vision Pipeline:
Camera → Preprocessing → YOLOv8n → Face Embedding → Event Trigger

Audio Pipeline:
Mic → Whisper-small → Intent Classifier → Response Generator → On-device TTS

Automation Layer:
Handles access control, alerting, and visitor interaction logic.

⸻

🎯 Example Scenarios
	•	Hospitals & Offices – Automates visitor check-in, reduces load on security staff.
	•	Residential Complexes – 24/7 AI gatekeeper, instantly alerts residents.
	•	Events & Public Spaces – Provides multilingual guidance and detects crowd anomalies.

⸻

⚡ Getting Started

🔹 Requirements
	•	Snapdragon-powered Copilot+ PC or Galaxy S25 (hackathon kit)
	•	Python 3.10+
	•	Dependencies:

pip install -r requirements.txt



🔹 Running Athena
	1.	Clone this repo:

git clone https://github.com/ankithn30/AthenaSecurityGuardian.git
cd AthenaSecurityGuardian


	2.	Start the security guardian:

python main.py


	3.	Access the local web dashboard at:

http://localhost:5000



⸻

🧪 Demo
	1.	Point your phone camera (via QR connection) as the external video feed.
	2.	Ask Athena questions like:
	•	“Where is Room 203?”
	•	“Is this person allowed entry?”
	3.	Simulate threats (tailgating, loitering) and watch Athena respond in real-time.

⸻

🌍 Impact

Athena demonstrates the power of Snapdragon Edge AI to:
	•	Reduce costs by replacing manual monitoring
	•	Enhance safety and reliability without sacrificing privacy
	•	Scale security for residential, corporate, and public spaces
	•	Even potentially save lives in critical scenarios (e.g., hospitals, emergencies)

⸻

👥 Team

Athena Security Guardian was developed at the Snapdragon Multiverse Hackathon (Princeton, 2025) by:
	- Satvika Maram -satvika.maram@gmail.com
    - Ankith Nagabandi - ankithnagabandi@gmail.com

⸻
📜 License

This project is licensed under the MIT License.
