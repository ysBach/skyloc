<!-- Header block for project -->
<hr>

<div align="center">

<!-- [INSERT YOUR LOGO IMAGE HERE (IF APPLICABLE)] -->
<!-- ☝️ Replace with your logo (if applicable) via ![](https://uri-to-your-logo-image) ☝️ -->
<!-- ☝️ If you see logo rendering errors, make sure you're not using indentation, or try an HTML IMG tag -->

<h1 align="center">skyloc: Sky Locator</h1>
<!-- ☝️ Replace with your repo name ☝️ -->

</div>

<pre align="center">Convenience tools related to Sidereal or Solar System Objects (SSO) locator (e.g., apparent position and movements) for general purposes.</pre>
<!-- ☝️ Replace with a single sentence describing the purpose of your repo / proj ☝️ -->

<!-- Header block for project -->

<!-- [INSERT YOUR BADGES HERE (SEE: https://shields.io)] [![SLIM](https://img.shields.io/badge/Best%20Practices%20from-SLIM-blue)](https://nasa-ammos.github.io/slim/) -->
<!-- ☝️ Add badges via: https://shields.io e.g. ![](https://img.shields.io/github/your_chosen_action/your_org/your_repo) ☝️ -->

<!-- [INSERT SCREENSHOT OF YOUR SOFTWARE, IF APPLICABLE] -->
<!-- ☝️ Screenshot of your software (if applicable) via ![](https://uri-to-your-screenshot) ☝️ -->

This project is heavily dependent on [kete](https://github.com/Caltech-IPAC/kete/tree/main#) and SPICE toolkits.
<!-- ☝️ Replace with a more detailed description of your repository, including why it was made and whom its intended for.  ☝️ -->

Initially motivated by NASA's SPHEREx mission (and NEO Surveyor mission).
<!-- example links>
[Website](INSERT WEBSITE LINK HERE) | [Docs/Wiki](INSERT DOCS/WIKI SITE LINK HERE) | [Discussion Board](INSERT DISCUSSION BOARD LINK HERE) | [Issue Tracker](INSERT ISSUE TRACKER LINK HERE)
-->

## Features
This is a collection of convenience tools for "finding objects in the FOV" or "Finding FOVs that contain certain object(s)", and main functionality is dependent heavily on [kete](https://github.com/Caltech-IPAC/kete/tree/main#).

* **SSO Orbital Elements**
  - Query and manage SSO orbital elements from SBDB (Small-Body Database)
  - Convenient file management for orbital element data

* **JPL Horizons Integration**
  - Download and manage DE (Development Ephemeris) files
  - Query vectors and ephemeris data using `astroquery`

* **SPICE toolkits**
  - Some convenience tools for SPICE toolkits (such as meta kernel generations)

* **Field of View (FOV) Analysis**
  - Check which FOVs contain specific sidereal sky locations and/or SSOs (similar to [kete](https://github.com/Caltech-IPAC/kete/tree/main#))
  - Easy-to-use interface for FOVs (`FOVCollection` class)

* **SSO Position and Motion**
  - Calculate apparent positions and movements of SSOs (uses [kete](https://github.com/Caltech-IPAC/kete/tree/main#))
  - Compute SSO magnitudes and phase angles
  - Additional ephemeris-related convenience tools

## Code Style

This project uses a 92-character line length (inspired by [Julia's BlueStyle](https://github.com/JuliaDiff/BlueStyle)) for code and 79 characters for docstrings. While this differs from the standard Python PEP8 recommendations, it provides a good balance for modern displays.

To maintain consistent formatting, use the `black` formatter:

```bash
black **/*.py
