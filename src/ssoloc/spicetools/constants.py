import spiceypy as sp


# Some constants that will be widely used.
# Use SPICE's convert for exact compatibility.
# For AU: sp.convrt(1.0, 'AU', 'KM') is  149597870.6136889
# ^^^ DIFFERENT from the modern (IAU 2012) definition of 1au := 149,597,870.700 km !!
# CSPICE reference also says it "should not be used" (https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/convrt_c.html#Restrictions)
AU2KM = 149_597_870.700
KM2AU = 1.0 / AU2KM
D2R = sp.convrt(1.0, "DEGREES", "RADIANS")
R2D = sp.convrt(1.0, "RADIANS", "DEGREES")
