from ortools.sat.python import cp_model
import random
import numpy as np


class HospitalAssignmentProblem:
    def __init__(
        self,
        num_days,
        num_nurses,
        num_patients,
        num_shifts,
        max_treatment_duration,
        max_working_time,
    ):
        self.num_days = num_days
        self.num_nurses = num_nurses
        self.num_patients = num_patients
        self.num_shifts = num_shifts
        self.max_treatment_duration = max_treatment_duration
        self.max_working_time = max_working_time
        self.model = cp_model.CpModel()

    def split_nurses_per_day(self):
        all_nurses = list(range(self.num_nurses))
        random.shuffle(all_nurses)
        splitted_nurses = [list(n) for n in np.array_split(all_nurses, self.num_shifts)]
        return splitted_nurses

    # (day, patient, shift, treatment_duration)
    def generate_patients_data(self):
        patient_data = {}
        for d in range(self.num_days):
            for p in range(self.num_patients):
                shift = random.randint(1, self.num_shifts)
                treatment_duration = random.randint(1, self.max_treatment_duration) * 10
                patient_data[d, p] = [shift, treatment_duration]
        return patient_data

    # (day, nurse, shift)
    def generate_nurses_data(self):
        nurses_data = {}
        all_nurses = list(range(self.num_nurses))
        for d in range(self.num_days):
            splitted_nurses = self.split_nurses_per_day()
            for s, nurses in enumerate(splitted_nurses):
                for n in nurses:
                    nurses_data[d, n] = s + 1
        return nurses_data

    def solve(self):
        patient_data = self.generate_patients_data()
        nurses_data = self.generate_nurses_data()

        # (day, nurse, patient) if shifts correspond
        assignments = {}
        for d in range(self.num_days):
            for n in range(self.num_nurses):
                n_shift = nurses_data[d, n]
                for p in range(self.num_patients):
                    p_shift = patient_data[d, p][0]
                    if n_shift == p_shift:
                        assignments[d, n, p] = self.model.NewBoolVar(f"{d}-{n}-{p}")

        # each day, each patient is assigned to at most one nurse
        for d in range(self.num_days):
            for p in range(self.num_patients):
                potential_nurses = [
                    l[1] for l in assignments.keys() if (l[0] == d and l[2] == p)
                ]
                self.model.AddAtMostOne(assignments[d, n, p] for n in potential_nurses)

        # each day, every nurse should not work more than max_working_time
        for d in range(self.num_days):
            for n in range(self.num_nurses):
                working_time = [
                    assignments[d, n, p] * patient_data[d, p][1]
                    for p in range(self.num_patients)
                    if (d, n, p) in assignments.keys()
                ]
                self.model.Add(sum(working_time) < self.max_working_time)

        # objective is to treat as many patients as possible
        objective = sum(assignments.values())
        self.model.Maximize(objective)

        solver = cp_model.CpSolver()
        status = solver.Solve(self.model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution = {}
            for k in assignments.keys():
                if solver.BooleanValue(assignments[k]):
                    d, n, p = k
                    if (d, n) not in solution.keys():
                        solution[d, n] = []
                    solution[d, n].append(p)

            print("Found solution:")
            for (d, n), p in solution.items():
                print(f"On day {d}, nurse {n} will be assigned to patients: {p}.")
            print()

            # check how many patients were treated each day
            for d in range(self.num_days):
                treated_patients = []
                nurses = [k[1] for k in solution.keys() if k[0] == d]
                for n in nurses:
                    treated_patients.extend(solution[d, n])
                print(
                    f"On day {d}, number of treated patients: {len(treated_patients)} / {self.num_patients}"
                )

        else:
            print("No solution found.")


def main():
    num_nurses = 50
    num_shifts = 3
    num_days = 20
    num_patients = 500
    max_treatment_duration = 6  # in 10 minutes chunks
    max_working_time = 8 * 60  # in minutes

    problem = HospitalAssignmentProblem(
        num_days,
        num_nurses,
        num_patients,
        num_shifts,
        max_treatment_duration,
        max_working_time,
    )
    problem.solve()


if __name__ == "__main__":
    main()
