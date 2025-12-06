import pygame
import math
import random
import numpy as np
import json
import datetime
from typing import Dict, Any, List

class SimulationResult:
    def __init__(self):
        self.goals_reached: int = 0
        self.caught: bool = False
        self.time_elapsed: float = 0.0
        self.distance_trace: List[float] = []
        self.goal_times: List[float] = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'goals_reached': self.goals_reached,
            'caught': self.caught,
            'time_elapsed': self.time_elapsed,
            'distance_trace': self.distance_trace,
            'goal_times': self.goal_times
        }

def run_switching_filter(seed: int = None, max_goals: int = 10, max_time: float = 30.0) -> Dict[str, Any]:
    """Run the Switching Filter with the updated rule:
    Entering the evasion threshold counts as immediate failure.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    result = SimulationResult()
    
    # Settings
    EVASION_THRESHOLD = 40   # entering this radius = failure
    w, h = 800, 600
    evader_speed = 8
    pursuer_speed = 1
    evader_radius, pursuer_radius, goal_radius = 15, 15, 20
    
    # Colors
    WHITE = (255, 255, 255)
    BLUE = (50, 150, 255)
    RED = (255, 80, 80)
    GREEN = (50, 200, 50)
    BLACK = (0, 0, 0)
    GRAY = (200, 200, 200)
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Switching Filter (Failure on entering unsafe zone)")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    
    # Initial positions
    evader_x = w // 4
    evader_y = h // 2
    pursuer_x = 3 * w // 4
    pursuer_y = h // 2
    
    # Initialize first goal
    goal_x, goal_y = random.randint(0, w), random.randint(0, h)
    goal_spawn_time = pygame.time.get_ticks() / 1000.0
    goal_counter = 0
    
    running = True
    game_over = False
    start_time = pygame.time.get_ticks() / 1000.0
    
    print("\n=== Switching Filter Simulation ===")
    print(f"Starting positions — Evader: ({evader_x}, {evader_y}), Pursuer: ({pursuer_x}, {pursuer_y})")
    
    while running and not game_over:
        current_time = pygame.time.get_ticks() / 1000.0 - start_time
        
        # Process inputs
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
        
        # ----------------------------
        # 1. Distance & failure check
        # ----------------------------
        pursuer_dx = pursuer_x - evader_x
        pursuer_dy = pursuer_y - evader_y
        dist_to_pursuer = math.sqrt(pursuer_dx**2 + pursuer_dy**2)
        result.distance_trace.append(dist_to_pursuer)
        
        # Instant failure rule
        if dist_to_pursuer < EVASION_THRESHOLD:
            result.caught = True
            result.time_elapsed = current_time
            game_over = True
            print(f"\n⚠ FAILURE: Evader entered unsafe radius ({EVASION_THRESHOLD}) at {current_time:.2f}s!")
            break
        
        # ----------------------------
        # 2. Evader movement: ALWAYS go toward the goal
        # ----------------------------
        goal_dx = goal_x - evader_x
        goal_dy = goal_y - evader_y
        goal_dist = math.sqrt(goal_dx**2 + goal_dy**2)
        
        if goal_dist > 0:
            evader_x += (goal_dx / goal_dist) * evader_speed
            evader_y += (goal_dy / goal_dist) * evader_speed
        
        evader_x = max(evader_radius, min(w - evader_radius, evader_x))
        evader_y = max(evader_radius, min(h - evader_radius, evader_y))
        
        # ----------------------------
        # 3. Pursuer moves toward evader
        # ----------------------------
        chase_dx = evader_x - pursuer_x
        chase_dy = evader_y - pursuer_y
        chase_dist = math.sqrt(chase_dx**2 + chase_dy**2)
        
        if chase_dist > 0:
            pursuer_x += (chase_dx / chase_dist) * pursuer_speed
            pursuer_y += (chase_dy / chase_dist) * pursuer_speed
        
        pursuer_x = max(pursuer_radius, min(w - pursuer_radius, pursuer_x))
        pursuer_y = max(pursuer_radius, min(h - pursuer_radius, pursuer_y))
        
        # ----------------------------
        # 4. Goal reaching
        # ----------------------------
        goal_dist_now = math.sqrt((evader_x - goal_x)**2 + (evader_y - goal_y)**2)
        
        if goal_counter < max_goals and goal_dist_now <= evader_radius + goal_radius:
            goal_time = current_time - (goal_spawn_time - start_time)
            result.goal_times.append(goal_time)
            goal_counter += 1
            print(f"Goal {goal_counter} reached at {current_time:.2f}s — dist={dist_to_pursuer:.1f}")
            
            if goal_counter >= max_goals:
                result.time_elapsed = current_time
                result.goals_reached = goal_counter
                game_over = True
            else:
                goal_x, goal_y = random.randint(0, w), random.randint(0, h)
                goal_spawn_time = pygame.time.get_ticks() / 1000.0
        
        # ----------------------------
        # 5. Time-out condition
        # ----------------------------
        if current_time >= max_time:
            result.time_elapsed = current_time
            result.goals_reached = goal_counter
            game_over = True
            print(f"\n⏳ Time's up! {goal_counter} goals reached in {max_time:.1f}s.")
        
        # ----------------------------
        # 6. Draw everything
        # ----------------------------
        screen.fill(WHITE)
        
        pygame.draw.circle(screen, BLUE, (int(evader_x), int(evader_y)), evader_radius)
        pygame.draw.circle(screen, RED, (int(pursuer_x), int(pursuer_y)), pursuer_radius)
        pygame.draw.circle(screen, GREEN, (int(goal_x), int(goal_y)), goal_radius)
        
        pygame.draw.line(screen, BLACK, (evader_x, evader_y), (pursuer_x, pursuer_y), 1)
        
        # Unsafe zone circle
        pygame.draw.circle(screen, GRAY, (int(pursuer_x), int(pursuer_y)), EVASION_THRESHOLD, 1)
        
        screen.blit(font.render(f"Goals: {goal_counter}/{max_goals}", True, GREEN), (10, 10))
        screen.blit(small_font.render(f"Time: {current_time:.1f}s", True, BLACK), (10, 50))
        screen.blit(small_font.render(f"Distance: {dist_to_pursuer:.1f}", True, BLACK), (10, 80))
        screen.blit(small_font.render("Entering gray zone = FAILURE", True, RED), (10, 110))
        
        pygame.display.flip()
        clock.tick(120)
    
    pygame.quit()
    
    if not result.caught and not game_over:
        result.time_elapsed = pygame.time.get_ticks() / 1000.0 - start_time
        result.goals_reached = goal_counter
    
    return result.to_dict()

# ------------------------------------------------------
# Multi-trial runner
# ------------------------------------------------------
def run_multiple_trials(num_trials: int = 15, max_goals: int = 10, max_time: float = 30.0):
    all_results = []
    
    print(f"\nRunning {num_trials} Switching Filter trials...")
    print("==================================================")
    
    for trial in range(num_trials):
        print(f"\n=== Trial {trial + 1}/{num_trials} ===")
        result = run_switching_filter(
            seed=trial,
            max_goals=max_goals,
            max_time=max_time
        )
        all_results.append(result)
        
        print(f"  Goals reached: {result['goals_reached']}")
        print(f"  Status: {'FAILURE' if result['caught'] else 'Completed'}")
        print(f"  Time: {result['time_elapsed']:.2f}s")
    
    caught_count = sum(r['caught'] for r in all_results)
    completed = [r for r in all_results if not r['caught']]
    
    print("\n=== SUMMARY ===")
    print(f"Total trials: {num_trials}")
    print(f"Failures (entered unsafe zone): {caught_count}")
    print(f"Successful runs: {num_trials - caught_count}")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"switching_filter_results_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump({"trials": all_results}, f, indent=2)
    
    print(f"\nResults saved to {filename}")
    return all_results

if __name__ == "__main__":
    run_multiple_trials(num_trials=15)
