from ortools.sat.python import cp_model
import random
import numpy as np
import pandas as pd
import time

global_patients_mean = []
global_time = []


# find 95% confidence interval using t-student or normal ditribution
def find_95_confidence_interval(values):
    n = len(values)
    mean = np.mean(values)
    std = np.std(values)
    if n == 10:
        return [
            round(mean - 2.262 * std / np.sqrt(n), 3),
            round(mean + 2.262 * std / np.sqrt(n), 3),
        ]
    if n == 20:
        return [
            round(mean - 2.093 * std / np.sqrt(n), 3),
            round(mean + 2.093 * std / np.sqrt(n), 3),
        ]

    elif n >= 30:
        return [
            round(mean - 1.96 * std / np.sqrt(n), 3),
            round(mean + 1.96 * std / np.sqrt(n), 3),
        ]
    else:
        print("n should be 10, 20 or >=30")


class NursesAssignment:
    def __init__(
        self,
        num_nurses,
        num_shifts,
        num_days,
        max_shifts_worked,
        min_shifts_worked,
        min_nurses_per_shift,
    ):
        self.num_nurses = num_nurses
        self.num_shifts = num_shifts
        self.num_days = num_days
        self.max_shifts_worked = max_shifts_worked
        self.min_shifts_worked = min_shifts_worked
        self.min_nurses_per_shift = min_nurses_per_shift

    def solve(self):
        # define model
        model = cp_model.CpModel()

        # (day, nurse, shift) is True if on day d, nurse n in working on shift s
        shifts = {}
        for d in range(self.num_days):
            for n in range(self.num_nurses):
                for s in range(self.num_shifts):
                    shifts[(d, n, s)] = model.NewBoolVar(f"{d}-{n}-{s}")

        # each nurse is working at most 1 shift per day
        for d in range(self.num_days):
            for n in range(self.num_nurses):
                model.AddAtMostOne(shifts[(d, n, s)] for s in range(self.num_shifts))

        # there should be at least min_nurses_per_shift
        for d in range(self.num_days):
            for s in range(self.num_shifts):
                nurses_per_shift = []
                for n in range(self.num_nurses):
                    nurses_per_shift.append(shifts[(d, n, s)])
                model.Add(sum(nurses_per_shift) >= self.min_nurses_per_shift)

        # each nurse should work between min_shifts_worked and max_shifts_worked
        for n in range(self.num_nurses):
            shifts_worked = []
            for d in range(self.num_days):
                for s in range(self.num_shifts):
                    shifts_worked.append(shifts[d, n, s])
            model.Add(sum(shifts_worked) <= self.max_shifts_worked)
            model.Add(self.min_shifts_worked <= sum(shifts_worked))

        # objective is to maximize the number of shifts worked
        objective = sum(shifts.values())
        model.Maximize(objective)

        # solve the problem
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        # the solution returned is {[day, nurse]: shift}
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            nurses_assignment = {}
            for k in shifts.keys():
                if solver.BooleanValue(shifts[k]):
                    d, n, s = k
                    nurses_assignment[d, n] = s + 1
            # print("Solution found for nurse assignment.")
            # for (d, n), s in nurses_assignment.items():
            #     print(f"On day {d+1} nurse {n+1} will work shift {s}.")
            return nurses_assignment
        else:
            # print("Solution not found for nurse assignment.")
            return -1


class HospitalAssignment:
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

    def split_nurses_per_day(self):
        all_nurses = list(range(self.num_nurses))
        random.shuffle(all_nurses)
        splitted_nurses = [list(n) for n in np.array_split(all_nurses, self.num_shifts)]
        return splitted_nurses

    # {(day, patient): [shift, treatment_duration]}
    def generate_patients_data(self):
        patient_data = {}
        for d in range(self.num_days):
            for p in range(self.num_patients):
                shift = random.randint(1, self.num_shifts)
                treatment_duration = random.randint(1, self.max_treatment_duration) * 10
                patient_data[d, p] = [shift, treatment_duration]
        return patient_data

    def save_patients_assignment(self, assignment):
        index = [f"patient_{i+1}" for i in range(self.num_patients)]
        columns = [f"day_{i+1}" for i in range(self.num_days)]
        data = [["-"] * self.num_days] * self.num_patients
        df = pd.DataFrame(index=index, columns=columns, data=data)
        for (d, p), [s, t] in assignment.items():
            df.iloc[p, d] = s
        df.to_html("patients_assignment.html")

    def save_nurses_assignment(self, assignment):
        index = [f"nurse_{i+1}" for i in range(self.num_nurses)]
        columns = [f"day_{i+1}" for i in range(self.num_days)]
        data = [["-"] * self.num_days] * self.num_nurses
        df = pd.DataFrame(index=index, columns=columns, data=data)
        for (d, n), s in assignment.items():
            df.iloc[n, d] = s
        df.to_html("nurses_assignment.html")

    def save_hospital_assignment(self, assignment):
        index = [f"patient_{i+1}" for i in range(self.num_patients)]
        columns = [f"day_{i+1}" for i in range(self.num_days)]
        data = [["-"] * self.num_days] * self.num_patients
        df = pd.DataFrame(index=index, columns=columns, data=data)
        for (d, n), p_list in assignment.items():
            for p in p_list:
                df.iloc[p, d] = f"nurse_{n+1}"
        df.to_html("hospital_assignment.html")

    # {[day, nurse]: shift} randomly
    def generate_nurses_data_randomly(self):
        nurses_data = {}
        all_nurses = list(range(self.num_nurses))
        for d in range(self.num_days):
            splitted_nurses = self.split_nurses_per_day()
            for s, nurses in enumerate(splitted_nurses):
                for n in nurses:
                    nurses_data[d, n] = s + 1
        return nurses_data

    def generate_nurses_data_from_assignment(self, nurses_assignment):
        if nurses_assignment != -1:
            return nurses_assignment
        else:
            return -1

    def solve(self, assignment=None):
        start = time.time()
        patient_data = self.generate_patients_data()
        print("Patients data generated randomly. Check 'patients_assignment.html'.")
        self.save_patients_assignment(patient_data)

        nurses_data = self.generate_nurses_data_from_assignment(assignment)
        if not isinstance(nurses_data, dict):
            print("Nurses data generated randomly. Check 'nurses_assignment.html'.")
            nurses_data = self.generate_nurses_data_randomly()
        else:
            print(
                "Nurses data generated from assignment. Check 'nurses_assignment.html'."
            )
        self.save_nurses_assignment(nurses_data)

        # define the model
        model = cp_model.CpModel()

        # (day, nurse, patient) if shifts correspond
        assignments = {}
        for d in range(self.num_days):
            for n in range(self.num_nurses):
                if (d, n) in nurses_data.keys():
                    n_shift = nurses_data[d, n]
                    for p in range(self.num_patients):
                        p_shift = patient_data[d, p][0]
                        if n_shift == p_shift:
                            assignments[d, n, p] = model.NewBoolVar(f"{d}-{n}-{p}")

        # each day, each patient is assigned to at most one nurse
        for d in range(self.num_days):
            for p in range(self.num_patients):
                potential_nurses = [
                    l[1] for l in assignments.keys() if (l[0] == d and l[2] == p)
                ]
                model.AddAtMostOne(assignments[d, n, p] for n in potential_nurses)

        # each day, every nurse should not work more than max_working_time
        for d in range(self.num_days):
            for n in range(self.num_nurses):
                working_time = [
                    assignments[d, n, p] * patient_data[d, p][1]
                    for p in range(self.num_patients)
                    if (d, n, p) in assignments.keys()
                ]
                model.Add(sum(working_time) < self.max_working_time)

        # objective is to treat as many patients as possible
        objective = sum(assignments.values())
        model.Maximize(objective)

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution = {}
            for k in assignments.keys():
                if solver.BooleanValue(assignments[k]):
                    d, n, p = k
                    if (d, n) not in solution.keys():
                        solution[d, n] = []
                    solution[d, n].append(p)

            print(
                "Solution found for hospital assignment. Check 'hospital_assignment.html'."
            )
            self.save_hospital_assignment(solution)
            # for (d, n), p in solution.items():
            #     print(f"On day {d+1}, nurse {n+1} will be assigned to patients: {p}.")
            # print()

            # check how many patients were treated each day
            percentages = []
            for d in range(self.num_days):
                treated_patients = []
                nurses = [k[1] for k in solution.keys() if k[0] == d]
                for n in nurses:
                    treated_patients.extend(solution[d, n])
                # print(
                #     f"On day {d+1}, number of treated patients: {len(treated_patients)} / {self.num_patients}"
                # )
                percentages.append(len(treated_patients) / self.num_patients)

            end = time.time()
            global_time.append(end - start)
            global_patients_mean.append(np.mean(percentages))
            # print(f"Time for solving assignment problem: {end-start}")
            # print(f"Mean percentage of treated patients: {np.mean(percentages)}")

        else:
            print("No solution found for hospital assignment.")


def main():
    num_nurses = 100
    num_shifts = 3
    num_days = 5
    num_patients = 800
    max_treatment_duration = 10  # in 10 minutes chunks
    max_working_time = 8 * 60  # in minutes
    max_shifts_worked = 5
    min_shifts_worked = 3
    min_nurses_per_shift = 25  # less than num_nurses // num_shifts
    n_repetitions = 5
    print(f"Number of experiments: {n_repetitions}")
    for i in range(n_repetitions):
        nurses_assignment_problem = NursesAssignment(
            num_nurses,
            num_shifts,
            num_days,
            max_shifts_worked,
            min_shifts_worked,
            min_nurses_per_shift,
        )
        nurses_assignment = nurses_assignment_problem.solve()

        hospital_assignment_problem = HospitalAssignment(
            num_days,
            num_nurses,
            num_patients,
            num_shifts,
            max_treatment_duration,
            max_working_time,
        )
        hospital_assignment_problem.solve(nurses_assignment)

    print(
        f"95% confidence interval for treated patients percentage: {find_95_confidence_interval(global_patients_mean)}"
    )
    print(
        f"95% confidence interval for solving problem time: {find_95_confidence_interval(global_time)}"
    )


if __name__ == "__main__":
    main()
