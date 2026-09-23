import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def create_fuzzy_system():
    ph = ctrl.Antecedent(np.arange(0, 14.1, 0.1), "ph")
    turbidity = ctrl.Antecedent(np.arange(0, 21, 0.1), "turbidity")
    tds = ctrl.Antecedent(np.arange(0, 1001, 1), "tds")
    quality = ctrl.Consequent(np.arange(0, 101, 1), "quality")

    ph["acidic"] = fuzz.trapmf(ph.universe, [0, 0, 5.5, 6.5])
    ph["normal"] = fuzz.trimf(ph.universe, [6.0, 7.0, 8.0])
    ph["alkaline"] = fuzz.trapmf(ph.universe, [7.5, 8.5, 14, 14])

    turbidity["low"] = fuzz.trapmf(turbidity.universe, [0, 0, 2, 5])
    turbidity["medium"] = fuzz.trimf(turbidity.universe, [3, 7, 12])
    turbidity["high"] = fuzz.trapmf(turbidity.universe, [8, 13, 20, 20])

    tds["low"] = fuzz.trapmf(tds.universe, [0, 0, 250, 400])
    tds["medium"] = fuzz.trimf(tds.universe, [300, 500, 700])
    tds["high"] = fuzz.trapmf(tds.universe, [600, 750, 1000, 1000])

    quality["poor"] = fuzz.trapmf(quality.universe, [0, 0, 25, 45])
    quality["moderate"] = fuzz.trimf(quality.universe, [30, 50, 70])
    quality["good"] = fuzz.trimf(quality.universe, [55, 72, 88])
    quality["excellent"] = fuzz.trapmf(quality.universe, [80, 92, 100, 100])

    rules = [
        ctrl.Rule(ph["normal"] & turbidity["low"] & tds["low"], quality["excellent"]),
        ctrl.Rule(ph["normal"] & turbidity["low"] & tds["medium"], quality["good"]),
        ctrl.Rule(ph["normal"] & turbidity["medium"] & tds["medium"], quality["good"]),
        ctrl.Rule(ph["normal"] & turbidity["high"], quality["poor"]),
        ctrl.Rule(ph["acidic"] & turbidity["high"], quality["poor"]),
        ctrl.Rule(ph["alkaline"] & tds["high"], quality["moderate"]),
        ctrl.Rule(ph["acidic"] & tds["high"], quality["poor"]),
        ctrl.Rule(turbidity["high"] & tds["high"], quality["poor"]),
        ctrl.Rule(ph["normal"] & turbidity["low"] & tds["high"], quality["moderate"]),
    ]

    return ctrl.ControlSystem(rules)


def assess_water_quality(ph_value, turbidity_value, tds_value):
    system = create_fuzzy_system()
    simulation = ctrl.ControlSystemSimulation(system)

    simulation.input["ph"] = ph_value
    simulation.input["turbidity"] = turbidity_value
    simulation.input["tds"] = tds_value

    simulation.compute()

    score = simulation.output["quality"]

    if score < 40:
        category = "Poor"
    elif score < 60:
        category = "Moderate"
    elif score < 80:
        category = "Good"
    else:
        category = "Excellent"

    return round(score, 2), category
