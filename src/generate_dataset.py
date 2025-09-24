import json
import random
import operator
from itertools import combinations, permutations
from concurrent.futures import ThreadPoolExecutor, as_completed

class DatasetGenerator:
    """
    Generates a synthetic dataset for the Countdown math problem.
    Each sample consists of 6 numbers, a target value, and a valid solution expression.
    """
    def __init__(self):
        self.ops = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv
        }
        self.small_numbers = list(range(1, 11)) * 2
        self.large_numbers =  [25, 50, 75, 100]

    def generate_problem(self):
        """Generates a single solvable Countdown problem."""
        num_large = random.randint(0, 4)
        large_sample = random.sample(self.large_numbers, num_large)
        small_sample = random.sample(self.small_numbers, 6 - num_large)
        numbers = large_sample + small_sample
        random.shuffle(numbers)

        # Try to generate a solvable problem up to 100 times
        for _ in range(100):
            target, solution = self._generate_solution(numbers)
            if target is not None and 101 <= target <= 999 and target == int(target):
                return {
                    "numbers": numbers,
                    "target": int(target),
                    "solution": solution
                }
        return None

    def _generate_solution(self, numbers):
        """
        Attempts to find a random valid expression that results in a target number.
        This uses a brute-force approach on permutations of numbers and operations.
        """
        num_to_use = random.randint(3, 6) # Use a subset of numbers for variable difficulty
        for num_perm in permutations(numbers, num_to_use):
            for op_perm in permutations(self.ops.keys(), num_to_use - 1):
                expr_str = str(num_perm[0])
                result = float(num_perm[0])

                try:
                    for i in range(num_to_use - 1):
                        op_symbol = op_perm[i]
                        num = num_perm[i+1]
                        # Avoid division by zero
                        if op_symbol == '/' and num == 0:
                            raise ZeroDivisionError
                        # Apply operation
                        result = self.ops[op_symbol](result, num)
                        expr_str += f" {op_symbol} {num}"

                    # Check for valid target range and integer result
                    if 101 <= result <= 999 and result == int(result):
                        return result, expr_str
                except (ZeroDivisionError, OverflowError):
                    continue
        return None, None

    def generate_dataset(self, num_samples, output_path, max_workers=8):
        """Generates a dataset with a specified number of samples using parallelization."""
        dataset = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.generate_problem) for _ in range(num_samples * 2)]
            for future in as_completed(futures):
                problem = future.result()
                if problem:
                    dataset.append(problem)
                    if len(dataset) % 100 == 0:
                        print(f"Generated {len(dataset)}/{num_samples} samples...")
                    if len(dataset) >= num_samples:
                        break
        dataset = dataset[:num_samples]
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=4)
        print(f"Successfully generated dataset with {len(dataset)} samples at {output_path}")

if __name__ == "__main__":
    generator = DatasetGenerator()
    
    # Generate training dataset
    generator.generate_dataset(
        num_samples=5000, 
        output_path='../data/train_dataset.json'
    )
    
    # Generate testing dataset
    generator.generate_dataset(
        num_samples=500, 
        output_path='../data/test_dataset.json'
    )