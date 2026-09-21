# Tray Counter Application

This application is designed to record video streams from configured cameras and perform analysis, such as counting food trays. It fetches its configuration dynamically from an API and can be run as a persistent background service using systemd.

## Features

- **Dynamic Configuration**: Camera sources and other settings are fetched from a central API based on a device ID.
- **Scheduled Recording**: The recorder can be configured to run only during specific hours.
- **Automatic File Rotation & Retention**: Video files are automatically rotated at set intervals, and old recordings are cleaned up based on a retention policy.
- **Region of Interest (ROI) Setup**: Includes a one-time setup utility to capture a frame from each camera and upload it to Minio, facilitating ROI configuration via a dashboard.
- **Systemd Integration**: Provides a simple script to generate systemd service files, allowing the recorder and counter to run as manageable, persistent background services.

## Prerequisites

- Python 3.8+
- Access to a Minio-compatible object storage instance.
- Access to the configuration API.
- A Linux system with systemd (for running as a service).

## 1. Setup and Configuration

### a. Clone the Repository

```bash
git clone <your-repository-url>
cd tray_counter
```

### b. Create a Virtual Environment

It is highly recommended to use a Python virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate
```

### c. Install Dependencies

Install the required Python packages. (Assuming a `requirements.txt` file exists).

```bash
pip install -r requirements.txt
```

### d. Configure Environment Variables

The application is configured using a `.env` file. Copy the example file and edit it with your specific settings.

```bash
cp .env.example .env
nano .env
```

**Key variables to configure in your `.env` file:**

-   `API_BASE_URL`: The base URL of the configuration API.
-   `API_KEY`: Your authentication key for the API.
-   `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET`: Credentials for your Minio object storage.

## 2. Manual Usage

You can run the recorder and counter scripts directly from the command line for testing or manual operation.

### a. Running the Recorder

The `recorder.py` script has two modes of operation, determined by the `--setup-region` flag. It always requires an `EDGE_DEVICE_ID` as a command-line argument.

-   **To start the continuous recording process:**

    ```bash
    python recorder.py <YOUR_EDGE_DEVICE_ID>
    ```
    *(Replace `<YOUR_EDGE_DEVICE_ID>` with the actual ID, e.g., `AXELERA-M2-001`)*

-   **To perform the one-time ROI setup:**
    This will capture a single frame from each configured "tray" camera and upload it to Minio.

    ```bash
    python recorder.py <YOUR_EDGE_DEVICE_ID> --setup-region
    ```

### b. Running the Counter

To start the analysis process, run the `counter.py` script.

```bash
python counter.py
```

## 3. Running as a Systemd Service

For production environments, you can run the recorder and counter as background services using systemd. A helper script is provided to generate the necessary service files.

### a. Generate the Service Files

First, make the setup script executable:

```bash
chmod +x setup_systemd.sh
```

Now, run the script. It will automatically detect your project path and user, then create `recorder.service` and `counter.service` in your project directory.

```bash
./setup_systemd.sh
```

### b. Install and Start the Services

The script will print detailed instructions on how to install and manage the services. The steps are generally as follows:

1.  **Copy the files** to the systemd directory:
    ```bash
    sudo cp recorder.service counter.service /etc/systemd/system/
    ```

2.  **Reload systemd** to recognize the new services:
    ```bash
    sudo systemctl daemon-reload
    ```

3.  **Enable the services** to start automatically on boot:
    ```bash
    sudo systemctl enable recorder.service
    sudo systemctl enable counter.service
    ```

4.  **Start the services** immediately:
    ```bash
    sudo systemctl start recorder.service
    sudo systemctl start counter.service
    ```

### c. Managing the Services

You can manage the services using standard `systemctl` commands:

-   **Check the status:**
    ```bash
    sudo systemctl status recorder.service
    ```

-   **View live logs:**
    ```bash
    sudo journalctl -u recorder.service -f
    ```

-   **Stop a service:**
    ```bash
    sudo systemctl stop recorder.service
    ```

## Project Structure

```
.
├── recorder.py         # Main script for video recording & ROI setup
├── counter.py          # Main script for tray counting analysis
├── setup_systemd.sh    # Generates systemd service files
├── .env.example        # Example environment configuration file
├── tlabrecord/         # Python module for recording functionality
│   ├── orchestrator.py
│   └── camera_recorder.py
└── tlabcode/           # Python module for core logic and utilities
    └── setup_region_handler.py
```


```yolo export to onnx
yolo export model=best.pt imgsz=640 dynamic=True simplify=True opset=17 format=onnx
```