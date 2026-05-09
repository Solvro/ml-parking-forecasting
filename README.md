# ml-parking-forecasting
Forecasting the number of available PWr parking spots 

# Data
Unforutnately github doesn't suport large files so parquets with parking data can be downloaded form [Google Disk](https://drive.google.com/drive/folders/1ks8D8sBk5PMipJnqZ2SogV8AmlV_wfCo)

## Docker / Dev Container Setup

We recommend working with this repo through the **Dev Containers** extension in Visual Studio Code (or the equivalent in your IDE of choice). This gives you a reproducible environment with all dependencies preinstalled.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- [Visual Studio Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) (or an analogous extension in your IDE)

### Initial setup

1. Clone this repository and open a terminal in the project root (where `docker-compose.yaml` is located).
2. Build and start the container:
```bash
   docker-compose up --build
```
   The first build may take a few minutes while images are downloaded and dependencies installed.

### Day-to-day usage

After the initial build you don't need to run `docker-compose up --build` again unless dependencies change. Instead:

1. Open **Docker Desktop** and start the container.
2. In VS Code, open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) and select **Dev Containers: Attach to Running Container...**, then pick this project's container.
3. You're in — open a terminal inside the container and start working.

### Rebuilding

If `Dockerfile`, `docker-compose.yaml`, or dependency files (e.g. `requirements.txt`, `pyproject.toml`) change, rebuild with:
```bash
docker-compose up --build
```