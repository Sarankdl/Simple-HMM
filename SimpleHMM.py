import matplotlib.pyplot as plt
import numpy as np
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution

model = HiddenMarkovModel(name="Simple Model")

# emission probability distributions, P(umbrella | weather)
sunny_emissions = DiscreteDistribution({"yes": 0.1, "no": 0.9})
sunny_state = State(sunny_emissions, name="Sunny")

rainy_emissions = DiscreteDistribution({"yes": 0.8, "no": 0.2})
rainy_state = State(rainy_emissions, name="Rainy")

model.add_states(sunny_state, rainy_state)

model.add_transition(model.start, sunny_state, 0.5)
model.add_transition(model.start, rainy_state, 0.5)

model.add_transition(sunny_state, sunny_state, 0.8)
model.add_transition(sunny_state, rainy_state, 0.2)

model.add_transition(rainy_state, sunny_state, 0.4)
model.add_transition(rainy_state, rainy_state, 0.6)

model.bake()

print(model.dense_transition_matrix())