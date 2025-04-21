from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the network structure
model = BayesianNetwork([
    ('Disease', 'Fever'),
    ('Disease', 'Cough'),
    ('Disease', 'Fatigue'),
    ('Disease', 'Chills')
])

# Define CPDs
cpd_disease = TabularCPD('Disease', 2, [[0.3], [0.7]], state_names={'Disease': ['Flu', 'Cold']})

cpd_fever = TabularCPD(
    'Fever', 2,
    [[0.9, 0.5],   # Fever = Yes
     [0.1, 0.5]],  # Fever = No
    evidence=['Disease'], evidence_card=[2],
    state_names={'Fever': ['Yes', 'No'], 'Disease': ['Flu', 'Cold']}
)

cpd_cough = TabularCPD(
    'Cough', 2,
    [[0.8, 0.6],   # Cough = Yes
     [0.2, 0.4]],  # Cough = No
    evidence=['Disease'], evidence_card=[2],
    state_names={'Cough': ['Yes', 'No'], 'Disease': ['Flu', 'Cold']}
)

cpd_fatigue = TabularCPD(
    'Fatigue', 2,
    [[0.7, 0.3],   # Fatigue = Yes
     [0.3, 0.7]],  # Fatigue = No
    evidence=['Disease'], evidence_card=[2],
    state_names={'Fatigue': ['Yes', 'No'], 'Disease': ['Flu', 'Cold']}
)

cpd_chills = TabularCPD(
    'Chills', 2,
    [[0.6, 0.4],   # Chills = Yes
     [0.4, 0.6]],  # Chills = No
    evidence=['Disease'], evidence_card=[2],
    state_names={'Chills': ['Yes', 'No'], 'Disease': ['Flu', 'Cold']}
)

# Add CPDs to the model
model.add_cpds(cpd_disease, cpd_fever, cpd_cough, cpd_fatigue, cpd_chills)

# Validate model
assert model.check_model()

# Inference
infer = VariableElimination(model)

# --------------------
# Inference Task 1
# P(Disease | Fever=Yes, Cough=Yes)
result1 = infer.query(variables=['Disease'], evidence={'Fever': 'Yes', 'Cough': 'Yes'})
print("Inference Task 1: P(Disease | Fever=Yes, Cough=Yes)")
print(result1)

# --------------------
# Inference Task 2
# P(Disease | Fever=Yes, Cough=Yes, Chills=Yes)
result2 = infer.query(variables=['Disease'], evidence={'Fever': 'Yes', 'Cough': 'Yes', 'Chills': 'Yes'})
print("\nInference Task 2: P(Disease | Fever=Yes, Cough=Yes, Chills=Yes)")
print(result2)

# --------------------
# Inference Task 3
# P(Fatigue=Yes | Disease=Flu)
result3 = infer.query(variables=['Fatigue'], evidence={'Disease': 'Flu'})
print("\nInference Task 3: P(Fatigue=Yes | Disease=Flu)")
print(result3)
