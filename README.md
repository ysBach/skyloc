<!-- Header block for project -->
<hr>

<div align="center">

[INSERT YOUR LOGO IMAGE HERE (IF APPLICABLE)]
<!-- ☝️ Replace with your logo (if applicable) via ![](https://uri-to-your-logo-image) ☝️ -->
<!-- ☝️ If you see logo rendering errors, make sure you're not using indentation, or try an HTML IMG tag -->

<h1 align="center">ssoloc</h1>
<!-- ☝️ Replace with your repo name ☝️ -->

</div>

<pre align="center">Convenience tools related to Solar System Objects (SSO) locator (e.g., apparent position and movements) for general purposes.</pre>
<!-- ☝️ Replace with a single sentence describing the purpose of your repo / proj ☝️ -->

<!-- Header block for project -->

<!-- [INSERT YOUR BADGES HERE (SEE: https://shields.io)] [![SLIM](https://img.shields.io/badge/Best%20Practices%20from-SLIM-blue)](https://nasa-ammos.github.io/slim/) -->
<!-- ☝️ Add badges via: https://shields.io e.g. ![](https://img.shields.io/github/your_chosen_action/your_org/your_repo) ☝️ -->

<!-- [INSERT SCREENSHOT OF YOUR SOFTWARE, IF APPLICABLE] -->
<!-- ☝️ Screenshot of your software (if applicable) via ![](https://uri-to-your-screenshot) ☝️ -->

This project is heavily dependent on kete and SPICE toolkits.
<!-- ☝️ Replace with a more detailed description of your repository, including why it was made and whom its intended for.  ☝️ -->

Initially motivated by NASA's SPHEREx mission (and NEO Surveyor mission).
<!-- example links>
[Website](INSERT WEBSITE LINK HERE) | [Docs/Wiki](INSERT DOCS/WIKI SITE LINK HERE) | [Discussion Board](INSERT DISCUSSION BOARD LINK HERE) | [Issue Tracker](INSERT ISSUE TRACKER LINK HERE)
-->

## Features

* Calculate apparent locations and movements of SSOs (i.e., basically the same as kete).

<!-- ☝️ Replace with a bullet-point list of your features ☝️ -->


I adopted a 92-character line length, which is quite unorthodox for Python (92 is taken from [Julia](https://github.com/JuliaDiff/BlueStyle), and admittedly chosen without any solid reasoning). Personally, I find it wide enough without being excessive for most monitors. Following this, I used 79-character line length for docstrings. Contributors are free to revert to the more standard 79- and 72-character limits following PEP8, but to me, that feels too narrow---especially given that Python practically enforces 4-space indentation.

Other than this line length, I usually use `black` formatter to format all files by

    $ black **/*.py
