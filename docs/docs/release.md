---
icon: fontawesome/solid/truck-fast
title: Release
---

<!--
SPDX-FileCopyrightText: 2025 Delos Data Inc
SPDX-License-Identifier: Apache-2.0
-->

## Release process

Releases follow this workflow:

1. **Decide to release** — Maintainers decide when to cut a new release.

2. **Validate** — Ensure all of the following pass:

    - Unit tests
    - Security checks
    - System validation

3. **Create a version tag** — Create a new Git tag that follows [Semantic Versioning 2.0.0](https://semver.org/) (SemVer):

    - **MAJOR** version when making incompatible API changes
    - **MINOR** version when adding functionality in a backward compatible manner
    - **PATCH** version when making backward compatible bug fixes

    Version format: `MAJOR.MINOR.PATCH` (e.g. `1.2.3`). Pre-release and build metadata may be used as extensions to this format.

4. **Draft release** — Pushing the tag triggers a GitHub workflow that creates a **draft release** on GitHub.

5. **Review and edit** — Review the draft release on GitHub and edit release notes or details as needed.

6. **Publish** — Publish the release on GitHub to make it official.

## Release artifacts

Published artifacts are available as Docker images on [Docker Hub](https://hub.docker.com/) under the **openmosaic** organization.

| Image | Description |
|-------|-------------|
| `openmosaic/mosaic-vllm` | vLLM container with the Open Mosaic NCCL profiler plugin. |
| `openmosaic/mosaic-gpu-pcie-exporter` | Prometheus exporter for GPU-to-PCIe port mapping. Discovers GPUs via `nvidia-smi` (NVIDIA) or `rocm-smi` (AMD) and exposes metrics with host, gpu_id, gpu_uuid, pcie_port, and vendor. See [GPU PCIe Exporter](https://github.com/open-mosaic/mosaic/blob/main/deployments/gpu_pcie_exporter/README.md) for build, run, and configuration details. |
| `openmosaic/pipeline-analyzer` | Pipeline analysis for the NCCL profiler: analyses Prometheus metrics to identify pipeline parallelism structure and how communicators form pipeline stages. See [Pipeline Analyzer](https://github.com/open-mosaic/mosaic/blob/main/deployments/pipeline-analyzer/analyze_pipelines.py) for behavior and usage. |

Example (replace `1.0.0` with the desired version):

```bash
docker pull openmosaic/mosaic-vllm:1.0.0
docker pull openmosaic/mosaic-gpu-pcie-exporter:1.0.0
docker pull openmosaic/pipeline-analyzer:1.0.0
```
