# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-06-15

### Fixed
- Fixed `ImportError` by consuming public `pqfilt` v0.2.1 APIs (`to_pyarrow_expr` and `filter_df`).

## [0.3] - 2026-06-15

### Added
- Galactic coordinate (`glx_lon`, `glx_lat`) support in final ephemeris.
- `sun_target_pa` support for Horizons queries.
- Automated package publication and release workflows.

## [0.2]

### Added
- `utils.py`: `au2km`, `km2au`, `kmps2aupd`, `aupd2kmps`
