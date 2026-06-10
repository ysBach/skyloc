"""Shared hard-coded Horizons reference data for ephemeris tests.

Reference epochs are UTC calendar times. Calculations convert them to TDB,
because kete propagation and SPICE states are evaluated in TDB.

Earth-target ephemerides use a Sun-centered observer because observing Earth
from the geocenter is singular. The other representative targets use a
geocentric observer.
"""

import numpy as np
import pandas as pd
from astropy.time import Time

import kete


AU_KM = 149_597_870.7
VEL_ATOL_AU_PER_DAY = 86_400.0 / AU_KM  # 1 km/s
DIST_ATOL_AU = 100.0 / AU_KM  # 100 km
ANGLE_ATOL_DEG = 0.1 / 3600.0  # 0.1 arcsec
SKYMOT_ATOL_DEG = 0.1  # 0.1 arcsec/minute

QUERY_DIST_ATOL_AU = DIST_ATOL_AU / 10.0  # 10 km
QUERY_VEL_ATOL_AU_PER_DAY = VEL_ATOL_AU_PER_DAY / 10.0  # 0.1 km/s
QUERY_ANGLE_ATOL_DEG = ANGLE_ATOL_DEG / 10.0  # 0.01 arcsec
QUERY_SKYMOT_ATOL_DEG = SKYMOT_ATOL_DEG / 10.0  # 0.01 arcsec/minute
QUERY_MAG_ATOL = 0.01  # Horizons magnitude models/rounding are imperfect.

REFERENCE_TIMES_UTC = (
    "2025-01-01T00:00:00",
    "2025-01-02T00:00:00",
    "2025-12-31T00:00:00",
)

TARGET_KEYS = (
    "earth",
    "4",
    "15",
    "433",
    "C/2025 A6",
    "saturn_barycenter",
)

# Compatibility literal used to assemble `HORIZONS_REFERENCES` below.
# Prefer the canonical nested database for new tests.
HORIZONS_REFERENCE_METADATA = {
    "queried_at": "2026-05-06",
    "reference_frame": "J2000 ecliptic",
    "vector_mode": "geometric",
    "ephemeris_mode": "light-time corrected Horizons observer ephemeris",
    "targets": {
        "earth": {
            "horizons_command": "399",
            "horizons_target": "Earth (399)",
            "source": "DE441",
            "solution_date": None,
            "observation_arc": None,
        },
        "4": {
            "horizons_command": "4;",
            "horizons_target": "4 Vesta (A807 FA)",
            "source": "JPL#36",
            "solution_date": "2021-Apr-13_11:15:57",
            "observation_arc": "1950-2021",
        },
        "15": {
            "horizons_command": "15;",
            "horizons_target": "15 Eunomia (A851 OA)",
            "source": "JPL#107",
            "solution_date": "2025-Nov-19_21:53:58",
            "observation_arc": "1851-2025",
        },
        "433": {
            "horizons_command": "433;",
            "horizons_target": "433 Eros (A898 PA)",
            "source": "JPL#659",
            "solution_date": "2021-May-24_17:55:05",
            "observation_arc": "1893-2021",
        },
        "C/2025 A6": {
            "horizons_command": "2025 A6",
            "horizons_target": "Lemmon (C/2025 A6)",
            "source": "JPL#34",
            "solution_date": "2026-May-05_09:12:15",
            "observation_arc": "2024-11-12 to 2026-05-04",
        },
        "saturn_barycenter": {
            "horizons_command": "6",
            "horizons_target": "Saturn Barycenter (6)",
            "source": "DE441",
            "solution_date": None,
            "observation_arc": None,
        },
    },
}

# Reference context for the compatibility literals below:
# - UTC strings are the user-facing reference epochs. `tdb_jd` stores the
#   corresponding TDB epoch used by kete/SPICE.
# - State vectors are Sun-centered, J2000 ecliptic, in AU and AU/day.
# - Earth ephemerides are computed from the Sun (because a geocentric Earth
#   target is singular).
# - (4) Vesta, (15) Eunomia, (433) Eros, C/2025 A6 and saturn_barycenter
#   ephemerides (major asteroid, standard asteroid, near-Earth asteroid,
#   comet with non-grav term, and a planet) are geocentric.
#   They're chosen because their orbital elements are well-known and very
#   unlikely to be significantly updated in the near future.
#   Distances in AU, angles in degrees, and rates in arcsec/minute.
# - `vmag` is skyloc's selected apparent magnitude. For C/2025 A6 it is the
#   comet total magnitude (`Tmag`) because the comet total-magnitude model is
#   brighter/available; `Nmag` is not available for the chosen orbit row.

# Orbit-element rows used to test kete/SSOLocator.propagate_n_body().
SMALL_BODY_ORBITS = pd.DataFrame(
    [
        {
            "desig": "4",
            "ecc": 0.09010712502848492,
            "incl": 7.143943095399415,
            "peri_dist": 2.148618722440708,
            "peri_arg": 151.5834106572105,
            "lon_node": 103.7029211881615,
            "peri_time": 2460901.9286610126,
            "epoch": 2460800.5,
            "H": 3.25,
            "G": 0.32,
            "M1": np.nan,
            "M2": np.nan,
            "K1": np.nan,
            "K2": np.nan,
            "PC": np.nan,
            "A1": 0.0,
            "A2": 0.0,
            "A3": 0.0,
            "DT": 0.0,
        },
        {
            "desig": "15",
            "ecc": 0.1877168034130067,
            "incl": 11.76051087304953,
            "peri_dist": 2.147065847984489,
            "peri_arg": 98.6343858821964,
            "lon_node": 292.8795548057483,
            "peri_time": 2460505.2596148853,
            "epoch": 2460800.5,
            "H": 5.42,
            "G": 0.23,
            "M1": np.nan,
            "M2": np.nan,
            "K1": np.nan,
            "K2": np.nan,
            "PC": np.nan,
            "A1": 0.0,
            "A2": 0.0,
            "A3": 0.0,
            "DT": 0.0,
        },
        {
            "desig": "433",
            "ecc": 0.2227480169011467,
            "incl": 10.82830761253864,
            "peri_dist": 1.133356943989735,
            "peri_arg": 178.9225697371719,
            "lon_node": 304.2718959654088,
            "peri_time": 2461088.844738645,
            "epoch": 2460800.5,
            "H": 10.41,
            "G": 0.46,
            "M1": np.nan,
            "M2": np.nan,
            "K1": np.nan,
            "K2": np.nan,
            "PC": np.nan,
            "A1": 0.0,
            "A2": 0.0,
            "A3": 0.0,
            "DT": 0.0,
        },
        {
            "desig": "C/2025 A6",
            "ecc": 0.9955647766012586,
            "incl": 143.663669078271,
            "peri_dist": 0.5298913851285814,
            "peri_arg": 132.9712696280756,
            "lon_node": 108.097838797584,
            "peri_time": 2460988.039747382,
            "epoch": 2460961.5,
            "H": np.nan,
            "G": np.nan,
            "M1": 10.4,
            "M2": np.nan,
            "K1": 9.75,
            "K2": np.nan,
            "PC": np.nan,
            "A1": 2.088995456696e-8,
            "A2": 3.370448946953e-9,
            "A3": 0.0,
            "DT": 0.0,
        },
    ]
)

# Geometric Sun-centered J2000 ecliptic vectors used to test live Horizons
# vectors and Kete state restoration. The nested `ephem` values are a legacy
# skyloc/kete ephemeris view kept for older helper tests.
REFERENCE = {
    "2025-01-01T00:00:00": {
        "tdb_jd": 2460676.5008007395,
        "state": {
            "earth": {
                "pos": [-0.17869721746879524, 0.9669802380923771, -5.1094234812809614e-05],
                "vel": [-0.017204693516447128, -0.0031937753751638204, 5.361872388578645e-09],
            },
            "4": {
                "pos": [-2.2505805472698572, -0.17568000220729432, 0.27926416660995995],
                "vel": [0.0018743355079829892, -0.011459904920941223, 0.00011203339500879747],
            },
            "15": {
                "pos": [0.09484179885321625, 2.295009942174382, 0.20392597001076196],
                "vel": [-0.011623755128311514, 0.0023363023227739774, -0.002039888837351184],
            },
            "433": {
                "pos": [-0.01618925047148386, -1.6885616636024756, -0.1844303950314119],
                "vel": [0.011758890145914429, -0.0021318371522871136, 0.0016289577437606046],
            },
            "C/2025 A6": {
                "pos": [-1.863099329497508, 4.1512378835451207, -0.3544949308392238],
                "vel": [0.0071059555472042, -0.008195474433223, 0.0030998492292423],
            },
            "saturn_barycenter": {
                "pos": [9.461069575829592, -1.7646098604779425, -0.3458769353840593],
                "vel": [0.0007091092589850708, 0.005475773226429316, -0.0001235570112563828],
            },
        },
        "ephem": {
            "earth": {
                "ra": 101.3878882364296,
                "dec": 23.023363427345924,
                "hel_ecl_lon": 100.47009594689258,
                "hel_ecl_lat": -0.0029770422971822953,
                "obs_ecl_lon": 100.47009594689258,
                "obs_ecl_lat": -0.0029770422971822953,
                "r_hel": 0.9833531812135968,
                "r_obs": 0.9833531812135968,
                "sky_motion": 2.5489232412494998,
                "sky_motion_pa": 94.50470052124882,
            },
            "4": {
                "ra": 209.23242077569776,
                "dec": -4.773296501204644,
                "hel_ecl_lon": -175.5365505686654,
                "hel_ecl_lat": 7.052177463700431,
                "obs_ecl_lon": -151.12291277706052,
                "obs_ecl_lat": 6.7325729806612635,
                "r_hel": 2.2746350779141253,
                "r_obs": 2.3825175701366867,
                "sky_motion": 0.9937995533330785,
                "sky_motion_pa": 104.89741570971763,
            },
            "15": {
                "ra": 76.47637637107135,
                "dec": 31.449706485250395,
                "hel_ecl_lon": 87.6335855920452,
                "hel_ecl_lat": 5.073444594704881,
                "obs_ecl_lon": 78.36135238255594,
                "obs_ecl_lat": 8.555182461943282,
                "r_hel": 2.3060033397056916,
                "r_obs": 1.3711648810820654,
                "sky_motion": 0.5514956895853347,
                "sky_motion_pa": -129.90091630554227,
            },
            "433": {
                "ra": 273.9340278739791,
                "dec": -27.355869219045264,
                "hel_ecl_lon": -90.54931207163324,
                "hel_ecl_lat": -6.233044923587047,
                "obs_ecl_lon": -86.49810697178674,
                "obs_ecl_lat": -3.9643877238106695,
                "r_hel": 1.6986810042589606,
                "r_obs": 2.6668909534431577,
                "sky_motion": 1.5588166629217122,
                "sky_motion_pa": 85.12564957409462,
            },
            "C/2025 A6": {
                "ra": 118.81334163379447,
                "dec": 15.07298551134821,
                "hel_ecl_lon": 114.1725294766056,
                "hel_ecl_lat": -4.456067506507647,
                "obs_ecl_lon": 117.88022145962819,
                "obs_ecl_lat": -5.621051133894122,
                "r_hel": 4.563748482671677,
                "r_obs": 3.619531197505188,
                "sky_motion": 0.7603038734706054,
                "sky_motion_pa": -74.29932018144623,
            },
            "saturn_barycenter": {
                "ra": 346.195126260193,
                "dec": -8.049656539351997,
                "hel_ecl_lon": -10.564999917011486,
                "hel_ecl_lat": -2.058219320033325,
                "obs_ecl_lon": -15.820970200471056,
                "obs_ecl_lat": -1.9768314161309883,
                "r_hel": 9.630436975174854,
                "r_obs": 10.025282230849776,
                "sky_motion": 0.18903858816010438,
                "sky_motion_pa": 65.57502267420395,
            },
        },
    },
    "2025-01-02T00:00:00": {
        "tdb_jd": 2460677.50080074,
        "state": {
            "earth": {
                "pos": [-0.19587281703872447, 0.9636354294613345, -5.114620201157605e-05],
                "vel": [-0.017145554570083304, -0.0034956409719691314, -1.0676462577170406e-07],
            },
            "4": {
                "pos": [-2.2486779132720134, -0.18713765259475723, 0.2793726870557348],
                "vel": [0.0019309363942902808, -0.011455346842206447, 0.00010500556620893133],
            },
            "15": {
                "pos": [0.0832169512838179, 2.2973185737257644, 0.2018836291306021],
                "vel": [-0.011625892504588436, 0.0022809715266077467, -0.0020447829405926153],
            },
            "433": {
                "pos": [-0.004429991119281594, -1.690642567376361, -0.18279589240284744],
                "vel": [0.011759510090499267, -0.0020300040409841377, 0.0016400251637765045],
            },
            "C/2025 A6": {
                "pos": [-1.855990458166515, 4.1430359352101132, -0.3513945285854319],
                "vel": [0.0071117902019688, -0.0082084329927463, 0.0031009549446445],
            },
            "saturn_barycenter": {
                "pos": [9.461777115013286, -1.7591338006909751, -0.34600043498332456],
                "vel": [0.0007059690280550031, 0.005476346046816479, -0.0001234421785989196],
            },
        },
        "ephem": {
            "earth": {
                "ra": 102.49160023204784,
                "dec": 22.93945727296048,
                "hel_ecl_lon": 101.48966395682112,
                "hel_ecl_lat": -0.002980107301499282,
                "obs_ecl_lon": 101.48966395682112,
                "obs_ecl_lat": -0.002980107301499282,
                "r_hel": 0.9833409398493231,
                "r_obs": 0.9833409398493231,
                "sky_motion": 2.5489048566409904,
                "sky_motion_pa": 94.9360564169123,
            },
            "4": {
                "ra": 209.61701486123007,
                "dec": -4.874677730613787,
                "hel_ecl_lon": -175.24273937712263,
                "hel_ecl_lat": 7.057909557106254,
                "obs_ecl_lon": -150.72563546668886,
                "obs_ecl_lat": 6.771265261209877,
                "r_hel": 2.273680267084357,
                "r_obs": 2.3698871551402383,
                "sky_motion": 0.9882553350604028,
                "sky_motion_pa": 104.7375532378234,
            },
            "15": {
                "ra": 76.28193283306933,
                "dec": 31.308108124191996,
                "hel_ecl_lon": 87.9254525848234,
                "hel_ecl_lat": 5.0188590366675685,
                "obs_ecl_lon": 78.18070189107577,
                "obs_ecl_lat": 8.429943527326014,
                "r_hel": 2.3076730032388246,
                "r_obs": 1.3774539852808223,
                "sky_motion": 0.5394119278382556,
                "sky_motion_pa": -131.03978812790902,
            },
            "433": {
                "ra": 274.63266487456576,
                "dec": -27.3013507455281,
                "hel_ecl_lon": -90.15013180101393,
                "hel_ecl_lat": -6.170948754445725,
                "obs_ecl_lon": -85.8746203547248,
                "obs_ecl_lat": -3.9283768428117507,
                "r_hel": 1.700501765281531,
                "r_obs": 2.667440286558775,
                "sky_motion": 1.556431808962801,
                "sky_motion_pa": 84.83451220811732,
            },
            "C/2025 A6": {
                "ra": 118.508010558674,
                "dec": 15.155801264809304,
                "hel_ecl_lon": 114.13305666430442,
                "hel_ecl_lat": -4.427327088051424,
                "obs_ecl_lon": 117.57357580324792,
                "obs_ecl_lat": -5.5962785112622635,
                "r_hel": 4.553147898819195,
                "r_obs": 3.6037057915560684,
                "sky_motion": 0.7705520913782496,
                "sky_motion_pa": -74.3151695393268,
            },
            "saturn_barycenter": {
                "ra": 346.2652861424194,
                "dec": -8.018139155311783,
                "hel_ecl_lon": -10.532178657994008,
                "hel_ecl_lat": -2.0590192056992467,
                "obs_ecl_lon": -15.744670548565615,
                "obs_ecl_lat": -1.9746201080129107,
                "r_hel": 9.630134692927781,
                "r_obs": 10.040087419994743,
                "sky_motion": 0.19238550317193398,
                "sky_motion_pa": 65.6193466699023,
            },
        },
    },
    "2025-12-31T00:00:00": {
        "tdb_jd": 2461040.5008007395,
        "state": {
            "earth": {
                "pos": [-0.15706408092181112, 0.9707233423516862, -5.959167229829809e-05],
                "vel": [-0.017257022027381445, -0.0028167934705611805, 5.793517167829859e-08],
            },
            "4": {
                "pos": [1.0901135223705873, -1.9105757974170283, -0.07601931663357075],
                "vel": [0.010642810359022613, 0.005310483336122994, -0.0014536427859775337],
            },
            "15": {
                "pos": [-2.875459951857511, 0.5301184301192039, -0.5086694995747013],
                "vel": [-0.0027854523556660555, -0.008836987792543579, -0.0012497344013519163],
            },
            "433": {
                "pos": [0.17504749580842835, 1.1665214894215956, 0.1533088567283083],
                "vel": [-0.016952063929399023, 0.0006542853050094682, -0.002609204275896931],
            },
            "C/2025 A6": {
                "pos": [-0.4892313923348294, -0.9733199641549751, -0.5644468126996893],
                "vel": [-0.0170120122702257, -0.0047181527781883, -0.0129721166975449],
            },
            "saturn_barycenter": {
                "pos": [9.507804450879824, 0.25217819347264125, -0.38285703646124974],
                "vel": [-0.00045982580657010397, 0.005564767971713869, -7.862063089539208e-05],
            },
        },
        "ephem": {
            "earth": {
                "ra": 100.00120692080313,
                "dec": 23.117297287150308,
                "hel_ecl_lon": 99.19086675935102,
                "hel_ecl_lat": -0.0034721705250918846,
                "obs_ecl_lon": 99.19086675935102,
                "obs_ecl_lat": -0.0034721705250918846,
                "r_hel": 0.9833478206887987,
                "r_obs": 0.9833478206887987,
                "sky_motion": 2.5470095834429256,
                "sky_motion_pa": 93.9610500411546,
            },
            "4": {
                "ra": 295.51224082614755,
                "dec": -22.776082949223987,
                "hel_ecl_lon": -60.29229995152283,
                "hel_ecl_lat": -1.979300798493756,
                "obs_ecl_lon": -66.59450446994222,
                "obs_ecl_lat": -1.3859306689623083,
                "r_hel": 2.2010057487049774,
                "r_obs": 3.140558324056775,
                "sky_motion": 1.3165153327116503,
                "sky_motion_pa": 82.84397601306969,
            },
            "15": {
                "ra": 184.2483648524663,
                "dec": -13.250839103995157,
                "hel_ecl_lon": 169.55426806975686,
                "hel_ecl_lat": -9.868889160859748,
                "obs_ecl_lon": -170.79341032501563,
                "obs_ecl_lat": -10.463990198097106,
                "r_hel": 2.967834251518441,
                "r_obs": 2.8004451143023936,
                "sky_motion": 0.461776156001942,
                "sky_motion_pa": 137.62696490222808,
            },
            "433": {
                "ra": 19.65742646932015,
                "dec": 31.792431829008475,
                "hel_ecl_lon": 81.46590685229194,
                "hel_ecl_lat": 7.405153354575524,
                "obs_ecl_lon": 30.521701789418348,
                "obs_ecl_lat": 21.693146691193913,
                "r_hel": 1.1895030965165752,
                "r_obs": 0.4149179373792668,
                "sky_motion": 1.4690319512002699,
                "sky_motion_pa": 114.44410589519882,
            },
            "C/2025 A6": {
                "ra": 257.9679104732484,
                "dec": -39.00668364080134,
                "hel_ecl_lon": -116.69505898359456,
                "hel_ecl_lat": -27.39450981924986,
                "obs_ecl_lon": -99.70017090695858,
                "obs_ecl_lat": -15.971685723410618,
                "r_hel": 1.2268463534940233,
                "r_obs": 2.051263879713088,
                "sky_motion": 0.8366249206553894,
                "sky_motion_pa": -177.73290483714322,
            },
            "saturn_barycenter": {
                "ra": 356.9972218971354,
                "dec": -3.7658610110700756,
                "hel_ecl_lon": 1.5193160017720464,
                "hel_ecl_lat": -2.3051113781384474,
                "obs_ecl_lon": -4.251894585743509,
                "obs_ecl_lat": -2.261898303830551,
                "r_hel": 9.518850709397706,
                "r_obs": 9.699099172176973,
                "sky_motion": 0.14207212853926027,
                "sky_motion_pa": 63.25039483993995,
            },
        },
    },
}

# Locally computed ephemeris fields merged into the legacy `REFERENCE`
# ephemeris view; strict Horizons ephemerides are in `HORIZONS_EPHEM_REFERENCE`.
REFERENCE_EPHEM_EXTRA = {
    "2025-01-01T00:00:00": {
        "earth": {
            "alpha": 0.0,
            "racosdec_rate": 2.5410493462222044,
            "dec_rate": -0.2001946798632827,
            "sun_target_pa": -175.49581216051212,
            "vmag": 99.0,
        },
        "4": {
            "alpha": 24.236594191121448,
            "racosdec_rate": 0.96039564090651,
            "dec_rate": -0.2554951371999088,
            "sun_target_pa": -67.3103527097875,
            "vmag": 7.828684797928144,
        },
        "15": {
            "alpha": 9.841575458391063,
            "racosdec_rate": -0.4230826169007607,
            "dec_rate": -0.3537634731110717,
            "sun_target_pa": 104.72189748604097,
            "vmag": 8.509822862867383,
        },
        "433": {
            "alpha": 4.62895327694141,
            "racosdec_rate": 1.5531791009887752,
            "dec_rate": 0.13245402543555373,
            "sun_target_pa": -121.08149332139396,
            "vmag": 13.97398804178453,
        },
        "C/2025 A6": {
            "alpha": 3.8726751225921823,
            "racosdec_rate": -0.7319358226117828,
            "dec_rate": 0.20574725172409747,
            "sun_target_pa": -61.55548721553977,
            "vmag": 19.621648211552852,
            "Tmag": 29.26422809695745,
        },
        "saturn_barycenter": {
            "alpha": 5.253339900814978,
            "racosdec_rate": 0.17212029368764847,
            "dec_rate": 0.07816771913292084,
            "sun_target_pa": 68.2417074189968,
            "vmag": 99.0,
        },
    },
    "2025-01-02T00:00:00": {
        "earth": {
            "alpha": 0.0,
            "racosdec_rate": 2.539451849848608,
            "dec_rate": -0.21931773870006746,
            "sun_target_pa": -175.0648705878645,
            "vmag": 99.0,
        },
        "4": {
            "alpha": 24.337705061783552,
            "racosdec_rate": 0.9557429452028415,
            "dec_rate": -0.25140411685242403,
            "sun_target_pa": -67.45771523391593,
            "vmag": 7.818787451755294,
        },
        "15": {
            "alpha": 10.25971406152446,
            "racosdec_rate": -0.4068535010284952,
            "dec_rate": -0.35416868381470396,
            "sun_target_pa": 103.32570582065954,
            "vmag": 8.535370988421235,
        },
        "433": {
            "alpha": 4.813010436904139,
            "racosdec_rate": 1.5501108446466878,
            "dec_rate": 0.1401297443798062,
            "sun_target_pa": -119.79069032977974,
            "vmag": 13.983485959458049,
        },
        "C/2025 A6": {
            "alpha": 3.6211648665991194,
            "racosdec_rate": -0.7418593177977959,
            "dec_rate": 0.20831533338663114,
            "sun_target_pa": -60.326035212253814,
            "vmag": 19.60228627542743,
            "Tmag": 29.23009573064294,
        },
        "saturn_barycenter": {
            "alpha": 5.209943916518724,
            "racosdec_rate": 0.17522916020269716,
            "dec_rate": 0.07941613970331038,
            "sun_target_pa": 68.27422980944596,
            "vmag": 99.0,
        },
    },
    "2025-12-31T00:00:00": {
        "earth": {
            "alpha": 8.537736462515939e-07,
            "racosdec_rate": 2.540925390238918,
            "dec_rate": -0.1759431140718677,
            "sun_target_pa": -176.03923956159719,
            "vmag": 99.0,
        },
        "4": {
            "alpha": 6.327340331755139,
            "racosdec_rate": 1.3062604755809926,
            "dec_rate": 0.16400058292514208,
            "sun_target_pa": 85.60047536803708,
            "vmag": 7.858151741510058,
        },
        "15": {
            "alpha": 19.3498333120716,
            "racosdec_rate": 0.3112162459637956,
            "dec_rate": -0.34114757290669995,
            "sun_target_pa": -66.21581214006169,
            "vmag": 10.900959441201039,
        },
        "433": {
            "alpha": 51.08179533296582,
            "racosdec_rate": 1.3373558394104395,
            "dec_rate": -0.6078932738911256,
            "sun_target_pa": 74.44189917077439,
            "vmag": 10.249465719548775,
        },
        "C/2025 A6": {
            "alpha": 19.452661118197884,
            "racosdec_rate": -0.03309516749543895,
            "dec_rate": -0.835970075869995,
            "sun_target_pa": -123.86090254655096,
            "vmag": 12.825811883471564,
            "Tmag": 14.124368213503978,
        },
        "saturn_barycenter": {
            "alpha": 5.766785610317689,
            "racosdec_rate": 0.12686785989952246,
            "dec_rate": 0.06394556929288563,
            "sun_target_pa": 67.11952650124306,
            "vmag": 99.0,
        },
    },
}


def _to_tdb_jd(utc_iso):
    return Time(utc_iso, scale="utc", format="isot").tdb.jd


def _sun_observer(jd):
    return kete.State(
        desig="sun",
        jd=float(jd),
        pos=kete.Vector([0.0, 0.0, 0.0], frame=kete.Frames.Ecliptic),
        vel=kete.Vector([0.0, 0.0, 0.0], frame=kete.Frames.Ecliptic),
        center_id=10,
    )


def _wide_fov(observer):
    return kete.fov.ConeFOV(
        observer=observer,
        pointing=kete.Vector.from_lat_lon(0.0, 0.0),
        angle=180.0,
    )


def _state_arrays(state):
    return (
        np.array([state.pos[0], state.pos[1], state.pos[2]], dtype=float),
        np.array([state.vel[0], state.vel[1], state.vel[2]], dtype=float),
    )


def _angle_delta_deg(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


# Light-time corrected JPL Horizons observer ephemerides used by strict query,
# core, and propagation ephemeris tests.
HORIZONS_EPHEM_REFERENCE = {
    "earth": {
        "2025-01-01T00:00:00": {
            "ra": 101.381618263,
            "dec": 23.023817969,
            "racosdec_rate": 2.532058333333333,
            "dec_rate": -0.2927216666666667,
            "vmag": -4.063,
            "hel_ecl_lon": 100.464305,
            "hel_ecl_lat": -0.002977,
            "r_hel": 0.983353289341,
            "r_obs": 0.98335328934128,
            "obs_ecl_lon": 100.4643054,
            "obs_ecl_lat": -0.002977,
            "alpha": 0.0058,
            "sky_motion": 2.5489231,
            "sky_motion_pa": 96.594467,
        },
        "2025-01-02T00:00:00": {
            "ra": 102.485338137,
            "dec": 22.939955239,
            "racosdec_rate": 2.5323166666666665,
            "dec_rate": -0.290325,
            "vmag": -4.063,
            "hel_ecl_lon": 101.483874,
            "hel_ecl_lat": -0.00298,
            "r_hel": 0.983341027117,
            "r_obs": 0.98334102711709,
            "obs_ecl_lon": 101.4838736,
            "obs_ecl_lat": -0.0029801,
            "alpha": 0.0058,
            "sky_motion": 2.5489051,
            "sky_motion_pa": 96.540301,
        },
        "2025-12-31T00:00:00": {
            "ra": 99.994933176,
            "dec": 23.117696742,
            "racosdec_rate": 2.5298233333333333,
            "dec_rate": -0.29538499999999995,
            "vmag": -4.063,
            "hel_ecl_lon": 99.185081,
            "hel_ecl_lat": -0.003472,
            "r_hel": 0.983347964205,
            "r_obs": 0.98334796420546,
            "obs_ecl_lon": 99.1850806,
            "obs_ecl_lat": -0.0034722,
            "alpha": 0.0058,
            "sky_motion": 2.54701,
            "sky_motion_pa": 96.659779,
        },
    },
    "4": {
        "2025-01-01T00:00:00": {
            "ra": 209.229055078,
            "dec": -4.771923485,
            "racosdec_rate": 0.9606231666666667,
            "dec_rate": -0.2543666666666667,
            "vmag": 7.83,
            "hel_ecl_lon": 184.459408,
            "hel_ecl_lat": 7.052097,
            "r_hel": 2.274648138385,
            "r_obs": 2.38246425164458,
            "sun_target_pa": -67.31,
            "obs_ecl_lon": 209.2212184,
            "obs_ecl_lat": 6.7302153,
            "alpha": 24.2408,
            "sky_motion": 0.9937301,
            "sky_motion_pa": 104.83121,
        },
        "2025-01-02T00:00:00": {
            "ra": 209.613649303,
            "dec": -4.873305889,
            "racosdec_rate": 0.9559693333333333,
            "dec_rate": -0.2502683333333333,
            "vmag": 7.82,
            "hel_ecl_lon": 184.753238,
            "hel_ecl_lat": 7.057832,
            "r_hel": 2.273693234771,
            "r_obs": 2.36983380498293,
            "sun_target_pa": -67.457,
            "obs_ecl_lon": 209.6186292,
            "obs_ecl_lat": 6.7688868,
            "alpha": 24.3419,
            "sky_motion": 0.9881861,
            "sky_motion_pa": 104.67052,
        },
        "2025-12-31T00:00:00": {
            "ra": 295.507948671,
            "dec": -22.776277903,
            "racosdec_rate": 1.3058871666666667,
            "dec_rate": 0.16717383333333335,
            "vmag": 7.835,
            "hel_ecl_lon": 299.702089,
            "hel_ecl_lat": -1.978626,
            "r_hel": 2.200992884948,
            "r_obs": 3.14056934546379,
            "sun_target_pa": 85.60199999999998,
            "obs_ecl_lon": 293.7606213,
            "obs_ecl_lat": -1.3884594,
            "alpha": 6.3217,
            "sky_motion": 1.3165441,
            "sky_motion_pa": 82.704927,
        },
    },
    "15": {
        "2025-01-01T00:00:00": {
            "ra": 76.471726973,
            "dec": 31.449999332,
            "racosdec_rate": -0.42407,
            "dec_rate": -0.35260166666666665,
            "vmag": 8.507,
            "hel_ecl_lon": 87.631272,
            "hel_ecl_lat": 5.073877,
            "r_hel": 2.305990095672,
            "r_obs": 1.37116774822607,
            "sun_target_pa": 104.71800000000002,
            "obs_ecl_lon": 78.7120473,
            "obs_ecl_lat": 8.559415,
            "alpha": 9.8393,
            "sky_motion": 0.5515096,
            "sky_motion_pa": 230.25741,
        },
        "2025-01-02T00:00:00": {
            "ra": 76.277293946,
            "dec": 31.308393522,
            "racosdec_rate": -0.40783833333333336,
            "dec_rate": -0.353055,
            "vmag": 8.533,
            "hel_ecl_lon": 87.923132,
            "hel_ecl_lat": 5.019294,
            "r_hel": 2.307659651326,
            "r_obs": 1.37745755993969,
            "sun_target_pa": 103.322,
            "obs_ecl_lon": 78.5314251,
            "obs_ecl_lat": 8.4341841,
            "alpha": 10.2574,
            "sky_motion": 0.5394257,
            "sky_motion_pa": 229.11812,
        },
        "2025-12-31T00:00:00": {
            "ra": 184.245854495,
            "dec": -13.249590344,
            "racosdec_rate": 0.31124433333333335,
            "dec_rate": -0.34108,
            "vmag": 10.911,
            "hel_ecl_lon": 169.551353,
            "hel_ecl_lat": -9.868566,
            "r_hel": 2.967812548596,
            "r_obs": 2.8003753327283,
            "sun_target_pa": -66.21499999999997,
            "obs_ecl_lon": 189.5678272,
            "obs_ecl_lat": -10.4636263,
            "alpha": 19.353,
            "sky_motion": 0.4617449,
            "sky_motion_pa": 137.61871,
        },
    },
    "433": {
        "2025-01-01T00:00:00": {
            "ra": 273.929720989,
            "dec": -27.356576983,
            "racosdec_rate": 1.5528473333333332,
            "dec_rate": 0.13656095,
            "vmag": 13.926,
            "hel_ecl_lon": 269.444532,
            "hel_ecl_lat": -6.234,
            "r_hel": 1.698652846986,
            "r_obs": 2.66684888611341,
            "sun_target_pa": -121.07,
            "obs_ecl_lon": 273.8416018,
            "obs_ecl_lat": -3.968173,
            "alpha": 4.6348,
            "sky_motion": 1.5588405,
            "sky_motion_pa": 84.974207,
        },
        "2025-01-02T00:00:00": {
            "ra": 274.628372994,
            "dec": -27.30208044,
            "racosdec_rate": 1.5497581666666667,
            "dec_rate": 0.14422483333333333,
            "vmag": 13.935,
            "hel_ecl_lon": 269.843726,
            "hel_ecl_lat": -6.171907,
            "r_hel": 1.700473893845,
            "r_obs": 2.66739782544582,
            "sun_target_pa": -119.77999999999997,
            "obs_ecl_lon": 274.4651773,
            "obs_ecl_lat": -3.932158,
            "alpha": 4.8189,
            "sky_motion": 1.5564546,
            "sky_motion_pa": 84.683208,
        },
        "2025-12-31T00:00:00": {
            "ra": 19.654606262,
            "dec": 31.790345786,
            "racosdec_rate": 1.3368011666666668,
            "dec_rate": -0.6090883333333333,
            "vmag": 10.231,
            "hel_ecl_lon": 81.463944,
            "hel_ecl_lat": 7.405424,
            "r_hel": 1.189508352527,
            "r_obs": 0.41495201775414,
            "sun_target_pa": 74.44,
            "obs_ecl_lon": 30.8842195,
            "obs_ecl_lat": 21.6961773,
            "alpha": 51.0773,
            "sky_motion": 1.4690222,
            "sky_motion_pa": 114.4955,
        },
    },
    "C/2025 A6": {
        "2025-01-01T00:00:00": {
            "ra": 118.811955759,
            "dec": 15.074208511,
            "racosdec_rate": -0.7313266666666667,
            "dec_rate": 0.20734833333333333,
            "hel_ecl_lon": 114.171637,
            "hel_ecl_lat": -4.45542,
            "r_hel": 4.564165859504,
            "r_obs": 3.6199423028101,
            "sun_target_pa": -61.557000000000016,
            "obs_ecl_lon": 118.2333363,
            "obs_ecl_lat": -5.6172048,
            "alpha": 3.8707,
            "sky_motion": 0.7601527,
            "sky_motion_pa": 285.82928,
            "Tmag": 19.622,
        },
        "2025-01-02T00:00:00": {
            "ra": 118.506655389,
            "dec": 15.157014717,
            "racosdec_rate": -0.7412416666666667,
            "dec_rate": 0.20994400000000002,
            "hel_ecl_lon": 114.132168,
            "hel_ecl_lat": -4.426682,
            "r_hel": 4.553563685472,
            "r_obs": 3.60411578960999,
            "sun_target_pa": -60.327999999999975,
            "obs_ecl_lon": 117.9268376,
            "obs_ecl_lat": -5.5924362,
            "alpha": 3.6192,
            "sky_motion": 0.7703997,
            "sky_motion_pa": 285.81387,
            "Tmag": 19.603,
        },
        "2025-12-31T00:00:00": {
            "ra": 257.980250094,
            "dec": -39.001848041,
            "racosdec_rate": -0.030346666666666668,
            "dec_rate": -0.8361033333333334,
            "hel_ecl_lon": 243.322135,
            "hel_ecl_lat": -27.387434,
            "r_hel": 1.226711610083,
            "r_obs": 2.05125577031134,
            "sun_target_pa": -123.87200000000001,
            "obs_ecl_lon": 260.6681887,
            "obs_ecl_lat": -15.9689094,
            "alpha": 19.4419,
            "sky_motion": 0.8366543,
            "sky_motion_pa": 182.07866,
            "Tmag": 12.825,
        },
    },
    "saturn_barycenter": {
        "2025-01-01T00:00:00": {
            "ra": 346.193423736,
            "dec": -8.050308728,
            "racosdec_rate": 0.17208683333333333,
            "dec_rate": 0.07827903333333334,
            "hel_ecl_lon": 349.4331,
            "hel_ecl_lat": -2.058173,
            "r_hel": 9.630454905538,
            "r_obs": 10.0253284354868,
            "sun_target_pa": 68.24200000000002,
            "obs_ecl_lon": 344.5240576,
            "obs_ecl_lat": -1.977563,
            "alpha": 5.2514,
            "sky_motion": 0.1890542,
            "sky_motion_pa": 65.540113,
        },
        "2025-01-02T00:00:00": {
            "ra": 346.263583631,
            "dec": -8.018791742,
            "racosdec_rate": 0.17519533333333334,
            "dec_rate": 0.07952891666666666,
            "hel_ecl_lon": 349.465918,
            "hel_ecl_lat": -2.058973,
            "r_hel": 9.630152651218,
            "r_obs": 10.040133456163,
            "sun_target_pa": 68.274,
            "obs_ecl_lon": 344.6003463,
            "obs_ecl_lat": -1.9753459,
            "alpha": 5.208,
            "sky_motion": 0.1924012,
            "sky_motion_pa": 65.584592,
        },
        "2025-12-31T00:00:00": {
            "ra": 356.995528499,
            "dec": -3.766552407,
            "racosdec_rate": 0.12686893333333332,
            "dec_rate": 0.06397185,
            "hel_ecl_lon": 1.517435,
            "hel_ecl_lat": -2.305081,
            "r_hel": 9.518868397085,
            "r_obs": 9.69914735904204,
            "sun_target_pa": 67.12,
            "obs_ecl_lon": 356.1096842,
            "obs_ecl_lat": -2.2620278,
            "alpha": 5.7649,
            "sky_motion": 0.1420849,
            "sky_motion_pa": 63.241126,
        },
    },
}

HORIZONS_ANGLE_COLS = (
    "ra",
    "dec",
    "hel_ecl_lon",
    "hel_ecl_lat",
    "obs_ecl_lon",
    "obs_ecl_lat",
    "alpha",
    "sun_target_pa",
)
HORIZONS_DISTANCE_COLS = ("r_hel", "r_obs")
HORIZONS_RATE_COLS = ("racosdec_rate", "dec_rate", "sky_motion")
HORIZONS_PA_COLS = ("sky_motion_pa",)
HORIZONS_MAG_COLS = ("vmag", "Tmag", "Nmag")


def _orbit_record_by_target():
    """Return small-body orbit rows keyed by target designation."""
    return {
        str(row["desig"]): row
        for row in SMALL_BODY_ORBITS.to_dict(orient="records")
    }


def _legacy_ephemeris_by_target(target_key):
    """Return the older skyloc/kete ephemeris reference view for one target."""
    return {
        utc_iso: {
            **REFERENCE[utc_iso]["ephem"][target_key],
            **REFERENCE_EPHEM_EXTRA[utc_iso][target_key],
        }
        for utc_iso in REFERENCE_TIMES_UTC
    }


def _build_horizons_references():
    """Assemble the canonical nested Horizons fixture database."""
    orbit_records = _orbit_record_by_target()
    metadata = HORIZONS_REFERENCE_METADATA

    return {
        "queried_at": metadata["queried_at"],
        "settings": {
            "reference_frame": metadata["reference_frame"],
            "vector_mode": metadata["vector_mode"],
            "ephemeris_mode": metadata["ephemeris_mode"],
        },
        "epochs_utc": REFERENCE_TIMES_UTC,
        "tdb_jd_by_utc": {
            utc_iso: REFERENCE[utc_iso]["tdb_jd"]
            for utc_iso in REFERENCE_TIMES_UTC
        },
        "targets": {
            target_key: {
                **metadata["targets"][target_key],
                "orbit": orbit_records.get(target_key),
                "vectors": {
                    utc_iso: REFERENCE[utc_iso]["state"][target_key]
                    for utc_iso in REFERENCE_TIMES_UTC
                },
                "legacy_ephemerides": _legacy_ephemeris_by_target(target_key),
                "horizons_ephemerides": HORIZONS_EPHEM_REFERENCE[target_key],
            }
            for target_key in TARGET_KEYS
        },
        "columns": {
            "angles": HORIZONS_ANGLE_COLS,
            "distances": HORIZONS_DISTANCE_COLS,
            "rates": HORIZONS_RATE_COLS,
            "position_angles": HORIZONS_PA_COLS,
            "magnitudes": HORIZONS_MAG_COLS,
        },
    }


# Canonical fixture database:
# - `settings` records the Horizons frame and geometric/light-time modes.
# - `targets[*]` stores per-target Horizons/SBDB provenance and solution dates.
# - `targets[*]["orbit"]` stores orbit-element rows for propagate_n_body().
# - `targets[*]["vectors"]` stores geometric Sun-centered J2000 states.
# - `targets[*]["horizons_ephemerides"]` stores strict Horizons ephemerides.
# - `targets[*]["legacy_ephemerides"]` stores the older skyloc/kete view.
HORIZONS_REFERENCES = _build_horizons_references()


def reference_tdb_jd(utc_iso):
    """Return the TDB JD for a reference UTC epoch."""
    return HORIZONS_REFERENCES["tdb_jd_by_utc"][utc_iso]


def reference_state(utc_iso, target_key):
    """Return a geometric Sun-centered vector reference for one target/epoch."""
    return HORIZONS_REFERENCES["targets"][target_key]["vectors"][utc_iso]


def reference_horizons_ephemeris(target_key, utc_iso):
    """Return a strict light-time corrected Horizons ephemeris reference."""
    return HORIZONS_REFERENCES["targets"][target_key]["horizons_ephemerides"][utc_iso]
