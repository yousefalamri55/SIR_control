

# Roadmap / to-do list

- [X] 2-cohort code with fixed-point iteration
- [ ] Extend to include exposed, asymptomatic, etc.
- [ ] Extend to N cohorts
- [ ] Extend to multiple regions
- [ ] Some test in which iteration makes a difference
- [ ] Incorporate economic factors
- [ ] Deal more precisely with positivity constraint
- [ ] Deal with situation in which multiple groups should be dosed on a single timestep (perhaps unnecessary if the timestep is kept small enough)


# Notes on literature

## Optimal control of vaccination

- Libotte 2020: Determination of an Optimal Control Strategy for Vaccine Administration in COVID-19 Pandemic Treatment
    - Uses genetic algorithms???
    - Single cohort, simple SIR
    - Time-dependent vaccine availability
    - Wrong objective functions
- Biswas 2014: A SEIR MODEL FOR CONTROL OF INFECTIOUS DISEASES WITH CONSTRAINTS
    - One cohort, SEIR with birth and death
    - Based on Lenhart's earlier paper, but introduces time-dependent availability of vaccine
- Kar 2011: Stability analysis and optimal control of an SIR epidemic model with vaccination
    - One cohort, simple SIR with birth and death
    - No limit on vaccination
- Rosario de Pinho (conf. proc.)
    - Illustrates how to deal with constraints in general (see also Biswas for this) and L1 constraint in particular.
