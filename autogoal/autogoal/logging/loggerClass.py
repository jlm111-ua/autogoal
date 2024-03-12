import json
import autogoal.logging
import requests
from autogoal.search import Logger


class WebSocketLogger(Logger):
    def __init__(self, uri, ip_data) -> None:
        self.uri = uri
        self.logger = autogoal.logging.logger()
        self.ip_data = ip_data

    def send_message(self, message):
        url = f"{self.uri}/wsTrain"
        data = {'ip_data': self.ip_data, 'message': message}
        try:
            response = requests.post(url, json=data)
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error de conexión: {e}")

    def begin(self, generations, pop_size):
        self.send_message("Search starting")

    def sample_solution(self, solution):
        self.send_message(f"Evaluating pipeline: {repr(solution)}")

    def eval_solution(self, solution, fitness):
        self.send_message(f"📈 Fitness={fitness}")

    # def error(self, e: Exception, solution):
    #     self.send_message(f"⚠️ Error: {e}")

    def start_generation(self, generations, best_solutions, best_fns):
        bests = "\n".join(f"Best_{i}: {fn}" for i, fn in enumerate(best_fns))
        self.send_message(f"New generation - Remaining={generations}\n{bests}")

    def update_best(
        self,
        solution,
        fn,
        new_best_solutions,
        best_solutions,
        new_best_fns,
        best_fns,
        new_dominated_solutions,
    ):
        self.send_message(f"🔥 New Best found {solution} {fn}")
        self.send_message(f"🔥 {len(new_best_solutions)} optimal solutions so far. Improved: {new_best_fns}. Previous {best_fns}.")

    def end(self, best_solutions, best_fns):
        self.send_message("Search finished")

        if len(best_fns) == 0:
            self.send_message("No solutions found")
        else:
            for i, (best_solution, best_fn) in enumerate(zip(best_solutions, best_fns)):
                self.send_message(f"{i}🌟 Optimal Solution {best_fn or 0}")
                self.send_message(repr(best_solution))

        self.send_message("Search finished")