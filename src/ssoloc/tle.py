# Just in case other query services are not working, TLE might be used..

import re
import requests


__all__ = ["fetch_tle_from_n2yo"]


def fetch_tle_n2yo(norad_id: int) -> tuple[str, str]:
    """
    Scrape the latest Two-Line Element Set (TLE) for the given NORAD ID
    from the public N2YO page at https://www.n2yo.com/satellite/?s={norad_id}.
    Returns a tuple (line1, line2).
    """
    url = f"https://www.n2yo.com/satellite/?s={norad_id}"
    resp = requests.get(url)
    resp.raise_for_status()

    # Regex matches two lines starting with "1 63182U ..." and "2 63182  ..."
    pattern = re.compile(r"1\s+\d{5}U[^\n]*\n\s*2\s+\d{5}[^\n]*")
    match = pattern.search(resp.text)
    if not match:
        raise RuntimeError("TLE block not found on page")

    line1, line2 = [ln.strip() for ln in match.group().splitlines()]
    return line1, line2


if __name__ == "__main__":
    # Example usage
    try:
        # 63182 : SPHEREx
        l1, l2 = fetch_tle_n2yo(63182)
        print("TLE line 1:", l1)
        print("TLE line 2:", l2)
    except Exception as e:
        print(f"Error fetching TLE: {e}")
