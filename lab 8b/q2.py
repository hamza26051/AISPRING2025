from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the structure
model = BayesianNetwork([
    ('Intelligence', 'Grade'),
    ('StudyHours', 'Grade'),
    ('Difficulty', 'Grade'),
    ('Grade', 'Pass')
])

# CPDs
cpd_intelligence = TabularCPD('Intelligence', 2, [[0.7], [0.3]], state_names={'Intelligence': ['High', 'Low']})
cpd_study = TabularCPD('StudyHours', 2, [[0.6], [0.4]], state_names={'StudyHours': ['Sufficient', 'Insufficient']})
cpd_difficulty = TabularCPD('Difficulty', 2, [[0.4], [0.6]], state_names={'Difficulty': ['Hard', 'Easy']})

cpd_grade = TabularCPD(
    variable='Grade', variable_card=3,
    values=[
        # A
        [0.80, 0.60, 0.60, 0.40, 0.50, 0.30, 0.20, 0.10],
        # B
        [0.15, 0.30, 0.25, 0.40, 0.35, 0.40, 0.40, 0.30],
        # C
        [0.05, 0.10, 0.15, 0.20, 0.15, 0.30, 0.40, 0.60],
    ],
    evidence=['Intelligence', 'StudyHours', 'Difficulty'],
    evidence_card=[2, 2, 2],
    state_names={
        'Grade': ['A', 'B', 'C'],
        'Intelligence': ['High', 'Low'],
        'StudyHours': ['Sufficient', 'Insufficient'],
        'Difficulty': ['Easy', 'Hard']
    }
)

cpd_pass = TabularCPD(
    variable='Pass', variable_card=2,
    values=[
        [0.95, 0.80, 0.50],  # Pass = Yes
        [0.05, 0.20, 0.50]   # Pass = No
    ],
    evidence=['Grade'],
    evidence_card=[3],
    state_names={'Pass': ['Yes', 'No'], 'Grade': ['A', 'B', 'C']}
)

# Add CPDs to model
model.add_cpds(cpd_intelligence, cpd_study, cpd_difficulty, cpd_grade, cpd_pass)

# Validate model
assert model.check_model()

# Inference
infer = VariableElimination(model)

# Query 1: What is the probability that the student passes the exam, given:
# StudyHours = Sufficient, Difficulty = Hard
q1 = infer.query(variables=['Pass'], evidence={'StudyHours': 'Sufficient', 'Difficulty': 'Hard'})
print("Probability of Pass | StudyHours=Sufficient, Difficulty=Hard:")
print(q1)

# Query 2: What is the probability that the student has High Intelligence, given:
# Pass = Yes
q2 = infer.query(variables=['Intelligence'], evidence={'Pass': 'Yes'})
print("\nProbability of Intelligence | Pass=Yes:")
print(q2)
