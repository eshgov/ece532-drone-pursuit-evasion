import pygame
import math
import random
import numpy as np
import json
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
    """Run the original Switching Filter simulation from the notebook."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    result = SimulationResult()
    
    # Settings from the notebook
    EVASION_THRESHOLD = 40
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
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Original Switching Filter")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    
    # Initial positions (same as notebook)
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
    
    print("\n=== Original Switching Filter Simulation ===")
    print(f"Starting positions - Evader: ({evader_x}, {evader_y}), "
          f"Pursuer: ({pursuer_x}, {pursuer_y})")
    
    while running and not game_over:
        current_time = pygame.time.get_ticks() / 1000.0 - start_time
        
        # Check for exit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
        
        if not game_over:
            # Calculate distance to pursuer
            pursuer_dx = pursuer_x - evader_x
            pursuer_dy = pursuer_y - evader_y
            dist_to_pursuer = np.sqrt(pursuer_dx**2 + pursuer_dy**2)
            result.distance_trace.append(dist_to_pursuer)
            
            # Check for capture (same as notebook)
            if dist_to_pursuer <= evader_radius + pursuer_radius:
                result.caught = True
                result.time_elapsed = current_time
                game_over = True
                print(f"\nEvader caught at {current_time:.2f}s!")
                break
            
            # Evader movement (same as notebook)
            if dist_to_pursuer >= EVASION_THRESHOLD:
                # Go to goal
                goal_dx = goal_x - evader_x
                goal_dy = goal_y - evader_y
                goal_dist = np.sqrt(goal_dx**2 + goal_dy**2)
                if goal_dist > 0:
                    evader_x += (goal_dx / goal_dist) * evader_speed
                    evader_y += (goal_dy / goal_dist) * evader_speed
            else:
                # Evade pursuer
                evasion_dx = evader_x - pursuer_x
                evasion_dy = evader_y - pursuer_y
                evasion_dist = np.sqrt(evasion_dx**2 + evasion_dy**2)
                if evasion_dist > 0:
                    evader_x += (evasion_dx / evasion_dist) * evader_speed
                    evader_y += (evasion_dy / evasion_dist) * evader_speed
            
            # Keep evader in bounds
            evader_x = max(evader_radius, min(w - evader_radius, evader_x))
            evader_y = max(evader_radius, min(h - evader_radius, evader_y))
            
            # Pursuer chases evader (same as notebook)
            capture_distance_dx = evader_x - pursuer_x
            capture_distance_dy = evader_y - pursuer_y
            capture_distance = np.sqrt(capture_distance_dx**2 + capture_distance_dy**2)
            if capture_distance > 0:
                pursuer_x += (capture_distance_dx / capture_distance) * pursuer_speed
                pursuer_y += (capture_distance_dy / capture_distance) * pursuer_speed
            
            # Keep pursuer in bounds
            pursuer_x = max(pursuer_radius, min(w - pursuer_radius, pursuer_x))
            pursuer_y = max(pursuer_radius, min(h - pursuer_radius, pursuer_y))
            
            # Check goal collection (same as notebook)
            goal_distance_dx = evader_x - goal_x
            goal_distance_dy = evader_y - goal_y
            goal_distance = np.sqrt(goal_distance_dx**2 + goal_distance_dy**2)
            
            if goal_counter < max_goals and goal_distance <= evader_radius + goal_radius:
                goal_time = current_time - (goal_spawn_time - start_time)
                result.goal_times.append(goal_time)
                goal_counter += 1
                print(f"Goal {goal_counter} reached at {current_time:.2f}s - "
                      f"Distance to pursuer: {dist_to_pursuer:.1f}")
                
                if goal_counter >= max_goals:
                    result.time_elapsed = current_time
                    result.goals_reached = goal_counter
                    game_over = True
                    print(f"\nAll {max_goals} goals collected in {current_time:.2f}s!")
                else:
                    # Set new goal
                    goal_x, goal_y = random.randint(0, w), random.randint(0, h)
                    goal_spawn_time = pygame.time.get_ticks() / 1000.0
            
            # Check max time
            if current_time >= max_time:
                result.time_elapsed = current_time
                result.goals_reached = goal_counter
                game_over = True
                print(f"\nTime's up! Reached {goal_counter} goals in {max_time:.1f}s")
        
        # Draw everything (same as notebook)
        screen.fill(WHITE)
        
        # Draw evader, pursuer, and goal
        pygame.draw.circle(screen, BLUE, (int(evader_x), int(evader_y)), evader_radius)
        pygame.draw.circle(screen, RED, (int(pursuer_x), int(pursuer_y)), pursuer_radius)
        if goal_counter < max_goals:
            pygame.draw.circle(screen, GREEN, (int(goal_x), int(goal_y)), goal_radius)
        
        # Draw line between evader and pursuer
        pygame.draw.line(screen, BLACK, (int(evader_x), int(evader_y)), 
                        (int(pursuer_x), int(pursuer_y)), 1)
        
        # Draw HUD
        screen.blit(font.render(f"Goals: {goal_counter}/{max_goals}", True, GREEN), (10, 10))
        
        # Show current mode
        if dist_to_pursuer >= EVASION_THRESHOLD:
            mode_text = "GOAL-SEEKING"
            mode_color = GREEN
        else:
            mode_text = "EVASION"
            mode_color = RED
        
        screen.blit(font.render(f"Mode: {mode_text}", True, mode_color), (10, 50))
        screen.blit(small_font.render(f"Time: {current_time:.1f}s", True, BLACK), (10, 90))
        screen.blit(small_font.render(f"Distance to pursuer: {dist_to_pursuer:.1f}", 
                                    True, BLACK), (10, 120))
        screen.blit(small_font.render("ESC to quit", True, BLACK), (10, 150))
        
        # Draw evasion threshold (visual aid)
        pygame.draw.circle(screen, (200, 200, 200, 100), 
                         (int(evader_x), int(evader_y)), 
                         EVASION_THRESHOLD, 1)
        
        pygame.display.flip()
        clock.tick(120)  # Same as notebook
    
    # Clean up
    if not result.caught and not game_over:
        result.time_elapsed = pygame.time.get_ticks() / 1000.0 - start_time
        result.goals_reached = goal_counter
    
    pygame.quit()
    return result.to_dict()

def run_multiple_trials(num_trials: int = 15, max_goals: int = 10, max_time: float = 30.0):
    """Run multiple trials and collect results."""
    all_results = []
    
    print(f"\nRunning {num_trials} trials of Switching Filter simulation...")
    print("==================================================")
    
    for trial in range(num_trials):
        seed = trial  # Use trial number as seed for reproducibility
        print(f"\n=== Trial {trial + 1}/{num_trials} ===")
        
        # Run simulation with visualization for the first trial only
        result = run_switching_filter(
            seed=seed,
            max_goals=max_goals,
            max_time=max_time
        )
        
        all_results.append(result)
        
        # Print trial summary
        print(f"  Goals: {result['goals_reached']}/{max_goals}")
        print(f"  Status: {'Caught!' if result['caught'] else 'Completed'}")
        print(f"  Time: {result['time_elapsed']:.2f}s")
    
    # Calculate and print overall statistics
    completed_goals = [r['goals_reached'] for r in all_results]
    completion_times = [r['time_elapsed'] for r in all_results if r['goals_reached'] >= max_goals]
    caught_count = sum(1 for r in all_results if r['caught'])
    
    print("\n=== Overall Statistics ===")
    print(f"Total trials: {num_trials}")
    print(f"Trials completed successfully: {len(completion_times)}/{num_trials}")
    print(f"Times caught: {caught_count}")
    if completion_times:
        print(f"\nAverage completion time: {sum(completion_times)/len(completion_times):.2f}s")
        print(f"Fastest completion: {min(completion_times):.2f}s")
        print(f"Slowest completion: {max(completion_times):.2f}s")
    
    # Save all results to file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'switching_filter_results_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump({
            'trials': all_results,
            'summary': {
                'total_trials': num_trials,
                'successful_trials': len(completion_times),
                'caught_count': caught_count,
                'avg_completion_time': sum(completion_times)/len(completion_times) if completion_times else None,
                'min_completion_time': min(completion_times) if completion_times else None,
                'max_completion_time': max(completion_times) if completion_times else None
            }
        }, f, indent=2)
    
    print(f"\nDetailed results saved to '{filename}'")
    return all_results

if __name__ == "__main__":
    import datetime
    run_multiple_trials(num_trials=15)
