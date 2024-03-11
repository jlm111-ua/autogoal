import websockets
import json

import autogoal.logging

class WebSocketLogger(Logger):
    def __init__(self, uri,ip_data) -> None:
        self.uri = uri
        self.logger = autogoal.logging.logger()
        self.ip_data = ip_data

    async def send_message(self, message):
        async with websockets.connect(self.uri) as websocket:
            await websocket.send(self.ip_data)
            await websocket.send(message)

    def begin(self, generations, pop_size):
        asyncio.run(self.send_message("Search starting"))

    def sample_solution(self, solution):
        asyncio.run(self.send_message(f"Evaluating pipeline: {repr(solution)}"))

    def eval_solution(self, solution, fitness):
        asyncio.run(self.send_message(f"📈 Fitness={fitness}"))

    def error(self, e: Exception, solution):
        asyncio.run(self.send_message(f"⚠️Error: {e}"))

    def start_generation(self, generations, best_solutions, best_fns):
        bests = "\n".join(f"Best_{i}: {fn}" for i, fn in enumerate(best_fns))
        asyncio.run(self.send_message(f"New generation - Remaining={generations}\n{bests}"))

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
        asyncio.run(self.send_message(f"🔥 New Best found {solution} {fn}"))
        asyncio.run(self.send_message(f"🔥 {len(new_best_solutions)} optimal solutions so far. Improved: {new_best_fns}. Previous {best_fns}."))

    def end(self, best_solutions, best_fns):
        asyncio.run(self.send_message("Search finished"))

        if len(best_fns) == 0:
            asyncio.run(self.send_message("No solutions found"))
        else:
            for i, (best_solution, best_fn) in enumerate(zip(best_solutions, best_fns)):
                asyncio.run(self.send_message(f"{i}🌟 Optimal Solution {best_fn or 0}"))
                asyncio.run(self.send_message(repr(best_solution)))

        asyncio.run(self.send_message("Search finished"))