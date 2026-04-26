import unittest

import pandas as pd

from src.weather_features import calculate_weather_features


class WeatherFeatureTests(unittest.TestCase):
    def test_calculate_weather_features(self):
        df = pd.DataFrame(
            {
                "PRECTOTCORR": [0, 2, 3, 0, 5, 1, 4],
                "RH2M": [60, 65, 70, 75, 80, 85, 90],
                "T2M": [20, 21, 22, 23, 24, 25, 26],
                "T2M_MAX": [28, 31, 32, 29, 30, 33, 27],
            }
        )
        features = calculate_weather_features(df)
        self.assertEqual(features["rainfall_7d"], 15.0)
        self.assertEqual(features["wet_days"], 4.0)
        self.assertEqual(features["heat_stress_days"], 3.0)


if __name__ == "__main__":
    unittest.main()
