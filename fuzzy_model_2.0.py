import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd

class FuzzyAnomalyDetector:
    """
    A fuzzy logic-based system for detecting anomalies (fraud) in credit card transactions.
    Uses Time and Amount as input variables to determine fraud likelihood.
    """
    
    def __init__(self):
        """Initialize the fuzzy system with input/output variables and membership functions."""
        # Define input and output universes
        self.time_universe = np.linspace(0, 1, 100)   # Scaled [0,1] universe for Time
        self.amount_universe = np.linspace(0, 1, 100)   # Scaled [0,1] universe for Amount
        self.fraud_universe = np.linspace(0, 1, 100)    # Fraud likelihood universe
        
        # Create fuzzy variables
        self.time = ctrl.Antecedent(self.time_universe, 'time')
        self.amount = ctrl.Antecedent(self.amount_universe, 'amount')
        self.fraud = ctrl.Consequent(self.fraud_universe, 'fraud')
        
        # Define membership functions for Time
        self.time['low'] = fuzz.trimf(self.time_universe, [0, 0, 0.3])
        self.time['medium'] = fuzz.trimf(self.time_universe, [0.2, 0.5, 0.8])
        self.time['high'] = fuzz.trimf(self.time_universe, [0.7, 1, 1])
        
        # Define membership functions for Amount
        self.amount['small'] = fuzz.trimf(self.amount_universe, [0, 0, 0.3])
        self.amount['medium'] = fuzz.trimf(self.amount_universe, [0.2, 0.5, 0.8])
        self.amount['large'] = fuzz.trimf(self.amount_universe, [0.7, 1, 1])
        
        # Define membership functions for Fraud likelihood
        self.fraud['low'] = fuzz.trimf(self.fraud_universe, [0, 0, 0.4])
        self.fraud['medium'] = fuzz.trimf(self.fraud_universe, [0.3, 0.5, 0.7])
        self.fraud['high'] = fuzz.trimf(self.fraud_universe, [0.6, 1, 1])
        
        # Explicitly set the defuzzification method
        self.fraud.defuzzify_method = 'centroid'
        
        # Initialize the rule base and control system placeholders
        self.rules = []
        self.control_system = None
        self.simulation = None
    
    def plot_membership_functions(self):
        """Plot the membership functions for visualization."""
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(10, 10))
        
        # Plot Time membership functions
        self.time.view(ax=ax0)
        ax0.set_title('Time')
        # Plot Amount membership functions
        self.amount.view(ax=ax1)
        ax1.set_title('Amount')
        # Plot Fraud membership functions
        self.fraud.view(ax=ax2)
        ax2.set_title('Fraud Likelihood')
        
        plt.tight_layout()
        plt.show()
    
    def define_rule_base(self):
        """
        Define the fuzzy rule base for fraud detection.
        You can expand these rules based on domain expertise.
        """
        rule1 = ctrl.Rule(self.amount['large'] & self.time['high'], self.fraud['high'])
        rule2 = ctrl.Rule(self.amount['large'] & self.time['medium'], self.fraud['medium'])
        rule3 = ctrl.Rule(self.amount['medium'] & self.time['high'], self.fraud['medium'])
        rule4 = ctrl.Rule(self.amount['small'] & self.time['low'], self.fraud['low'])
        rule5 = ctrl.Rule(self.amount['medium'] & self.time['medium'], self.fraud['medium'])
        rule6 = ctrl.Rule(self.amount['small'] & self.time['high'], self.fraud['medium'])
        
        # Add rules to ensure all combinations have some output
        rule7 = ctrl.Rule(self.amount['large'] & self.time['low'], self.fraud['medium'])
        rule8 = ctrl.Rule(self.amount['medium'] & self.time['low'], self.fraud['low'])
        rule9 = ctrl.Rule(self.amount['small'] & self.time['medium'], self.fraud['low'])
        
        # Store the rules in the class attribute
        self.rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9]
        print("Rule base defined with {} rules.".format(len(self.rules)))
    
    def build_control_system(self):
        """
        Build the fuzzy control system based on the defined rules.
        """
        # Create control system and simulation instance
        try:
            self.control_system = ctrl.ControlSystem(self.rules)
            self.simulation = ctrl.ControlSystemSimulation(self.control_system)
            print("Fuzzy control system built successfully.")
        except Exception as e:
            print(f"Error building control system: {e}")
            raise
    
    def predict(self, time_value, amount_value):
        """
        Predict fraud likelihood for a given transaction.
        
        Parameters:
            time_value (float): Scaled time value (0 to 1)
            amount_value (float): Scaled transaction amount (0 to 1)
        
        Returns:
            float: Fraud likelihood score (0 to 1)
        """
        # Error handling and input validation
        if not (0 <= time_value <= 1) or not (0 <= amount_value <= 1):
            print(f"Warning: Input values out of range. Time: {time_value}, Amount: {amount_value}")
            # Clip values to valid range
            time_value = max(0, min(time_value, 1))
            amount_value = max(0, min(amount_value, 1))
        
        try:
            # Set the inputs to the simulation
            self.simulation.input['time'] = time_value
            self.simulation.input['amount'] = amount_value
            
            # Compute the fuzzy inference
            self.simulation.compute()
            
            # Return the defuzzified output
            return self.simulation.output['fraud']
        except Exception as e:
            print(f"Error in prediction: {e}")
            print(f"Input values - Time: {time_value}, Amount: {amount_value}")
            # Return a default value if computation fails
            return 0.5  # Medium risk as fallback

# Example usage: applying fuzzy logic on transactions from a preprocessed dataset
if __name__ == "__main__":
    # Create an instance of the fuzzy anomaly detector
    detector = FuzzyAnomalyDetector()
    
    # Visualize membership functions (optional)
    detector.plot_membership_functions()
    
    # Define fuzzy rules and build the control system
    detector.define_rule_base()
    detector.build_control_system()
    
    try:
        # Load the preprocessed dataset
        df = pd.read_csv("creditcard_preprocessed.csv")
        
        # Apply fuzzy logic to each transaction and store the fraud risk score
        fraud_risks = []
        for idx, row in df.iterrows():
            risk = detector.predict(row['Time'], row['Amount'])
            fraud_risks.append(risk)
            
            # Print progress every 100 rows
            if idx % 100 == 0:
                print(f"Processed {idx} transactions...")
        
        # Add the fraud risk scores as a new column
        df['FraudRisk'] = fraud_risks
        
        # Save the results to a new CSV file
        df.to_csv("creditcard_fuzzy_results.csv", index=False)
        print("Fuzzy anomaly detection applied. Results saved to creditcard_fuzzy_results.csv")
    
    except Exception as e:
        print(f"Error in main execution: {e}")