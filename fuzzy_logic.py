import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# =========================================================
# Create Fuzzy System
# =========================================================

def create_fuzzy_system():

    # -----------------------------------------------------
    # Input variables
    # -----------------------------------------------------

    ph = ctrl.Antecedent(
        np.arange(0, 14.1, 0.1),
        "ph"
    )

    turbidity = ctrl.Antecedent(
        np.arange(0, 20.1, 0.1),
        "turbidity"
    )

    tds = ctrl.Antecedent(
        np.arange(0, 1001, 1),
        "tds"
    )


    # -----------------------------------------------------
    # Output variable
    # -----------------------------------------------------

    quality = ctrl.Consequent(
        np.arange(0, 101, 1),
        "quality"
    )


    # =====================================================
    # pH Membership Functions
    # =====================================================

    ph["acidic"] = fuzz.trapmf(
        ph.universe,
        [0, 0, 5.5, 6.5]
    )

    ph["normal"] = fuzz.trimf(
        ph.universe,
        [6.0, 7.0, 8.0]
    )

    ph["alkaline"] = fuzz.trapmf(
        ph.universe,
        [7.5, 8.5, 14.0, 14.0]
    )


    # =====================================================
    # Turbidity Membership Functions
    # =====================================================

    turbidity["low"] = fuzz.trapmf(
        turbidity.universe,
        [0, 0, 2, 5]
    )

    turbidity["medium"] = fuzz.trimf(
        turbidity.universe,
        [3, 7, 12]
    )

    turbidity["high"] = fuzz.trapmf(
        turbidity.universe,
        [8, 13, 20, 20]
    )


    # =====================================================
    # TDS Membership Functions
    # =====================================================

    tds["low"] = fuzz.trapmf(
        tds.universe,
        [0, 0, 250, 400]
    )

    tds["medium"] = fuzz.trimf(
        tds.universe,
        [300, 500, 700]
    )

    tds["high"] = fuzz.trapmf(
        tds.universe,
        [600, 750, 1000, 1000]
    )


    # =====================================================
    # Water Quality Membership Functions
    # =====================================================

    quality["poor"] = fuzz.trapmf(
        quality.universe,
        [0, 0, 25, 45]
    )

    quality["moderate"] = fuzz.trimf(
        quality.universe,
        [30, 50, 70]
    )

    quality["good"] = fuzz.trimf(
        quality.universe,
        [55, 72, 88]
    )

    quality["excellent"] = fuzz.trapmf(
        quality.universe,
        [80, 92, 100, 100]
    )


    # =====================================================
    # Fuzzy Rules
    # =====================================================

    rules = [

        # Excellent
        ctrl.Rule(
            ph["normal"]
            & turbidity["low"]
            & tds["low"],
            quality["excellent"]
        ),

        # Good
        ctrl.Rule(
            ph["normal"]
            & turbidity["low"]
            & tds["medium"],
            quality["good"]
        ),

        ctrl.Rule(
            ph["normal"]
            & turbidity["medium"]
            & tds["low"],
            quality["good"]
        ),

        ctrl.Rule(
            ph["normal"]
            & turbidity["medium"]
            & tds["medium"],
            quality["good"]
        ),

        # Moderate
        ctrl.Rule(
            ph["normal"]
            & turbidity["low"]
            & tds["high"],
            quality["moderate"]
        ),

        ctrl.Rule(
            ph["normal"]
            & turbidity["medium"]
            & tds["high"],
            quality["moderate"]
        ),

        ctrl.Rule(
            ph["alkaline"]
            & tds["high"],
            quality["moderate"]
        ),

        ctrl.Rule(
            ph["alkaline"]
            & turbidity["medium"],
            quality["moderate"]
        ),

        # Poor
        ctrl.Rule(
            ph["normal"]
            & turbidity["high"],
            quality["poor"]
        ),

        ctrl.Rule(
            ph["acidic"]
            & turbidity["high"],
            quality["poor"]
        ),

        ctrl.Rule(
            ph["acidic"]
            & tds["high"],
            quality["poor"]
        ),

        ctrl.Rule(
            turbidity["high"]
            & tds["high"],
            quality["poor"]
        ),

        # General pH rules
        ctrl.Rule(
            ph["acidic"],
            quality["poor"]
        ),

        ctrl.Rule(
            ph["alkaline"]
            & turbidity["high"],
            quality["poor"]
        ),
    ]


    return ctrl.ControlSystem(rules)


# =========================================================
# Assess Water Quality
# =========================================================

def assess_water_quality(
    ph_value,
    turbidity_value,
    tds_value
):

    # Validate inputs
    if not 0 <= ph_value <= 14:
        raise ValueError(
            "pH must be between 0 and 14."
        )

    if not 0 <= turbidity_value <= 20:
        raise ValueError(
            "Turbidity must be between 0 and 20 NTU."
        )

    if not 0 <= tds_value <= 1000:
        raise ValueError(
            "TDS must be between 0 and 1000 mg/L."
        )


    # Create fuzzy system
    system = create_fuzzy_system()

    simulation = ctrl.ControlSystemSimulation(
        system
    )


    # Set inputs
    simulation.input["ph"] = ph_value
    simulation.input["turbidity"] = turbidity_value
    simulation.input["tds"] = tds_value


    # Run fuzzy inference
    simulation.compute()


    # Make sure the output exists
    if "quality" not in simulation.output:

        raise RuntimeError(
            "The fuzzy system could not produce "
            "a quality score for these parameters."
        )


    score = float(
        simulation.output["quality"]
    )


    # =====================================================
    # Convert score into category
    # =====================================================

    if score < 40:

        category = "Poor"

    elif score < 60:

        category = "Moderate"

    elif score < 80:

        category = "Good"

    else:

        category = "Excellent"


    return round(score, 2), category
