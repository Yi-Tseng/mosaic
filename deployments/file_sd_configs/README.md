<!--
SPDX-FileCopyrightText: 2025 Delos Data Inc
SPDX-License-Identifier: Apache-2.0
-->

# File Service Discovery Configs

This folder holds Prometheus [file-based service discovery](https://prometheus.io/docs/prometheus/latest/configuration/configuration/#file_sd_config) configs. Targets defined here are scraped by the `file-service-discovery` job in the main Prometheus config.

File service discovery is used to dynamically add and remove Prometheus scrape targets based on GPU platform type (NVIDIA/AMD), IP address of GPU servers, and connection specifics for each scrape target without having to restart the Prometheus service.

## Docker mount

The directory is bind-mounted into the `otel-lgtm` container in `docker-compose.yml`:

```yaml
volumes:
  - ./file_sd_configs/:/otel-lgtm/file_sd_configs/
```

Files in this folder on the host appear at `/otel-lgtm/file_sd_configs/` inside the container. The Prometheus config uses:

```yaml
- job_name: 'file-service-discovery'
  file_sd_configs:
    - files:
        - '/otel-lgtm/file_sd_configs/*.yaml'
      refresh_interval: 2m
```

Any valid `*.yaml` in this directory is included automatically. No container restart is required when you add or change files.

## Adding services dynamically

New scrape targets are added by placing one or more `*.yaml` files in this folder. Each file must be a **list of target groups**. Each target group has:

- **`targets`** – list of `"host:port"` strings (metrics are fetched from `http://<target>/metrics`).
- **`labels`** – key/value labels attached to every metric from those targets (e.g. `job`, `host`).

Example for a single host with several exporters:

```yaml
- targets:
  - "hostname:5053"
  labels:
    job: gpu_exporter-hostname
    host: hostname
    __scrape_interval__: "10s"
    __scrape_timeout__: "5s"
- targets:
  - "hostname:9100"
  labels:
    job: node_exporter-hostname
    host: hostname
- targets:
  - "hostname:9256"
  labels:
    job: process_exporter-hostname
    host: hostname
```

Prometheus re-reads the file SD configs every `refresh_interval` (2m). New or updated `*.yaml` files are picked up within that period; removing a file drops its targets.

## Generator script

`file-sd-config-generate.sh` builds a YAML file in this format for one host. Run it from this directory:

```bash
./file-sd-config-generate.sh HOST_NAME [OPTIONS]
```

Use `--help` for full usage. Options include `--ip-address` (use an IP for targets instead of the host name) and port overrides for each exporter.

Example – generate `node1.yaml` using an IP and custom ports:

# FIXME
```bash
./file-sd-config-generate.sh node1 --ip-address 10.0.0.1 --gpu-exporter-port 9199 --node-exporter-port 9191
```

The script writes to `<HOST_NAME>.yaml` in this directory by default (or use `-o` to set the path).
