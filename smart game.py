import pygame
import numpy as np
import random
import time
from collections import deque

# === Game Settings ===
CELL_SIZE = 25
GRID_WIDTH = 25  # Reduced from 35 for better performance
GRID_HEIGHT = 15  # Reduced from 23 for better performance
WIDTH = GRID_WIDTH * CELL_SIZE
HEIGHT = GRID_HEIGHT * CELL_SIZE + 130  # Extra space for control bar

# Colors
WHITE = (255, 255, 255)
BLACK = (122, 43, 13)
BLUE = (59, 121, 186)
GREEN = (50, 125, 50)
RED = (255, 50, 50)
GRAY = (200, 200, 200)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
DARK_GRAY = (50, 50, 50)
LIGHT_BLUE = (173, 216, 230)
BUTTON_COLOR = (240, 240, 240)
BUTTON_HOVER = (220, 220, 220)
BUTTON_BORDER = (180, 180, 180)
SLIDER_BG = (220, 220, 220)
SLIDER_FG = (100, 149, 237)

START = (0, 0)
END = (GRID_WIDTH - 1, GRID_HEIGHT - 1)


# === UI Components ===
class Button:
    def __init__(self, x, y, width, height, text, action=None, icon=None, draggable=False):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action = action
        self.icon = icon
        self.hovered = False
        self.active = False
        self.dragging = False
        self.drag_offset = (0, 0)
        self.draggable = draggable

    def draw(self, screen, font):
        color = BUTTON_HOVER if self.hovered else BUTTON_COLOR
        if self.active:
            color = LIGHT_BLUE

        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, BUTTON_BORDER, self.rect, 2)

        text_surf = font.render(self.text, True, BLACK)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def is_over(self, pos):
        return self.rect.collidepoint(pos)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.is_over(event.pos)
            if self.dragging and self.draggable:
                self.rect.x = event.pos[0] - self.drag_offset[0]
                self.rect.y = event.pos[1] - self.drag_offset[1]
                self.rect.y = max(HEIGHT - 130, min(self.rect.y, HEIGHT - 30))

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_over(event.pos) and event.button == 1:
                if self.draggable:
                    self.dragging = True
                    self.drag_offset = (event.pos[0] - self.rect.x, event.pos[1] - self.rect.y)
                if self.action:
                    return self.action()
                return True
            return False

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False

        return False


class Slider:
    def __init__(self, x, y, width, height, min_val, max_val, initial_val):
        self.rect = pygame.Rect(x, y, width, height)
        self.handle_width = 20
        self.handle_rect = pygame.Rect(x, y, self.handle_width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.dragging = False
        self.drag_offset = 0
        self.update_handle_position()

    def update_handle_position(self):
        val_range = self.max_val - self.min_val
        pos = ((self.value - self.min_val) / val_range) * (self.rect.width - self.handle_width)
        self.handle_rect.x = self.rect.x + pos

    def get_value_from_pos(self, pos_x):
        relative_pos = pos_x - self.rect.x
        percent = max(0, min(1, relative_pos / (self.rect.width - self.handle_width)))
        return self.min_val + percent * (self.max_val - self.min_val)

    def draw(self, screen, font):
        pygame.draw.rect(screen, SLIDER_BG, self.rect)
        pygame.draw.rect(screen, BUTTON_BORDER, self.rect, 1)
        pygame.draw.rect(screen, SLIDER_FG, self.handle_rect)
        pygame.draw.rect(screen, BUTTON_BORDER, self.handle_rect, 1)

        value_text = font.render(f"{int(self.value)}", True, BLACK)
        text_rect = value_text.get_rect(midright=(self.rect.right + 30, self.rect.centery))
        screen.blit(value_text, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.handle_rect.collidepoint(event.pos):
                self.dragging = True
                self.drag_offset = event.pos[0] - self.handle_rect.x
                return True

        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False

        elif event.type == pygame.MOUSEMOTION and self.dragging:
            new_x = event.pos[0] - self.drag_offset
            new_x = max(self.rect.x, min(new_x, self.rect.right - self.handle_width))
            self.handle_rect.x = new_x
            self.value = self.get_value_from_pos(new_x)
            return True

        return False


class ControlBar:
    def __init__(self, screen_width):
        self.rect = pygame.Rect(0, HEIGHT - 130, screen_width, 130)
        self.dragging = False
        self.drag_offset = (0, 0)
        self.font = pygame.font.SysFont(None, 24)

        # Create controls
        btn_width = 120
        btn_height = 30
        btn_spacing = 20
        total_buttons_width = (btn_width * 3) + (btn_spacing * 2)
        start_x = (screen_width - total_buttons_width) // 2

        self.speed_slider = Slider(20, HEIGHT - 110, 150, 20, 1, 20, 5)
        self.show_path_btn = Button(start_x, HEIGHT - 110, btn_width, btn_height, "Toggle Path", None, draggable=False)
        self.restart_btn = Button(start_x + btn_width + btn_spacing, HEIGHT - 110, btn_width, btn_height, "New Maze",
                                  None, draggable=False)
        self.algo_btn = Button(start_x + 2 * (btn_width + btn_spacing), HEIGHT - 110, btn_width, btn_height,
                               "Algorithm", None, draggable=False)
        self.buttons = [self.show_path_btn, self.restart_btn, self.algo_btn]

        # Status information
        self.algorithm = "ACO"
        self.time = 0.0
        self.warning = False

    def draw(self, screen):
        pygame.draw.rect(screen, GRAY, self.rect)
        pygame.draw.line(screen, BLACK, (0, HEIGHT - 130), (WIDTH, HEIGHT - 130), 2)

        handle_rect = pygame.Rect(WIDTH // 2 - 25, HEIGHT - 130, 50, 5)
        pygame.draw.rect(screen, DARK_GRAY, handle_rect)

        speed_label = self.font.render("Speed:", True, BLACK)
        screen.blit(speed_label, (20, HEIGHT - 135))

        status_rect = pygame.Rect(0, HEIGHT - 130, WIDTH, 25)
        pygame.draw.rect(screen, LIGHT_BLUE, status_rect)

        time_text = f"Time: {self.time:.2f} sec"
        algo_text = f"Algorithm: {self.algorithm}"

        algo_surf = self.font.render(algo_text, True, BLACK)
        time_surf = self.font.render(time_text, True, BLACK)

        screen.blit(algo_surf, (10, HEIGHT - 125))
        screen.blit(time_surf, (180, HEIGHT - 125))

        if self.warning:
            warn_text = self.font.render("⚠ Warning: Off optimal path!", True, RED)
            screen.blit(warn_text, (WIDTH - warn_text.get_width() - 10, HEIGHT - 125))

        self.speed_slider.draw(screen, self.font)
        for button in self.buttons:
            button.draw(screen, self.font)

    def update_status(self, algorithm, time, warning):
        self.algorithm = algorithm
        self.time = time
        self.warning = warning

    def handle_event(self, event):
        if self.speed_slider.handle_event(event):
            return {"type": "speed", "value": self.speed_slider.value}

        for button in self.buttons:
            if button.handle_event(event):
                if button == self.show_path_btn:
                    return {"type": "toggle_path"}
                elif button == self.restart_btn:
                    return {"type": "restart"}
                elif button == self.algo_btn:
                    return {"type": "algorithm"}

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos) and event.pos[1] < HEIGHT - 125:
                self.dragging = True
                self.drag_offset = (event.pos[0], event.pos[1])
                return {"type": "drag_start"}

        elif event.type == pygame.MOUSEBUTTONUP:
            if self.dragging:
                self.dragging = False
                return {"type": "drag_end"}

        return None


# === Maze Generation ===
def generate_maze():
    maze = np.zeros((GRID_HEIGHT, GRID_WIDTH))
    for _ in range(GRID_WIDTH * GRID_HEIGHT // 3):
        x, y = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
        if (x, y) != START and (x, y) != END:
            maze[y][x] = 1
    return maze


# === Pathfinding Algorithms ===
class ACO:
    def __init__(self, maze, alpha=1.0, beta=3.0, evaporation=0.5, Q=1.0):
        self.maze = maze
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.Q = Q
        self.pheromone = np.ones_like(maze) * 0.1
        self.distances = self.compute_distances()
        self.best_path = None
        self.best_length = float('inf')

    def compute_distances(self):
        dist = np.ones_like(self.maze, dtype=float)
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.maze[y][x] == 1:
                    dist[y][x] = float('inf')
        return dist

    def get_neighbors(self, pos):
        x, y = pos
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and self.maze[ny][nx] == 0:
                neighbors.append((nx, ny))
        return neighbors

    def run(self, n_ants=20, n_iterations=3):
        for _ in range(n_iterations):
            all_paths = []
            for _ in range(n_ants):
                path = self.construct_path()
                if path:
                    all_paths.append(path)
                    if len(path) < self.best_length:
                        self.best_length = len(path)
                        self.best_path = path
            self.evaporate_pheromones()
            self.update_pheromones(all_paths)

    def construct_path(self):
        pos = START
        visited = set()
        path = [pos]
        while pos != END and len(path) < GRID_WIDTH * GRID_HEIGHT:
            visited.add(pos)
            neighbors = self.get_neighbors(pos)
            neighbors = [n for n in neighbors if n not in visited]
            if not neighbors:
                return None
            probs = []
            for n in neighbors:
                px, py = n
                tau = self.pheromone[py][px] ** self.alpha
                eta = (1.0 / (self.distances[py][px] + 1e-6)) ** self.beta
                probs.append(tau * eta)
            probs = np.array(probs)
            probs /= probs.sum()
            pos = neighbors[np.random.choice(len(neighbors), p=probs)]
            path.append(pos)
        return path if pos == END else None

    def evaporate_pheromones(self):
        self.pheromone *= (1 - self.evaporation)

    def update_pheromones(self, paths):
        for path in paths:
            contribution = self.Q / len(path)
            for (x, y) in path:
                self.pheromone[y][x] += contribution


class PSO:
    def __init__(self, maze, num_particles=10, max_iter=20, w=0.7, c1=1.5, c2=1.5):
        self.maze = maze
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.best_path = None
        self.best_length = float('inf')
        self.particles = []
        for _ in range(num_particles):
            particle = {
                'position': [START],
                'velocity': [],
                'best_position': None,
                'best_score': float('inf')
            }
            self.particles.append(particle)

    def get_neighbors(self, pos):
        x, y = pos
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and self.maze[ny][nx] == 0:
                neighbors.append((nx, ny))
        return neighbors

    def evaluate_path(self, path):
        if path[-1] != END:
            return float('inf')
        for x, y in path:
            if self.maze[y][x] == 1:
                return float('inf')
        return len(path)

    def run(self):
        for _ in range(self.max_iter):
            for particle in self.particles:
                current_pos = particle['position'][-1]
                if current_pos == END:
                    continue

                neighbors = self.get_neighbors(current_pos)
                if not neighbors:
                    continue

                if not particle['velocity']:
                    next_pos = random.choice(neighbors)
                    particle['velocity'] = [next_pos[0] - current_pos[0], next_pos[1] - current_pos[1]]

                vx, vy = particle['velocity']
                px, py = current_pos

                if particle['best_position'] and len(particle['position']) < len(particle['best_position']):
                    pb_pos = particle['best_position'][len(particle['position'])]
                    vx += self.c1 * random.random() * (pb_pos[0] - px)
                    vy += self.c1 * random.random() * (pb_pos[1] - py)

                if self.best_path and len(particle['position']) < len(self.best_path):
                    gb_pos = self.best_path[len(particle['position'])]
                    vx += self.c2 * random.random() * (gb_pos[0] - px)
                    vy += self.c2 * random.random() * (gb_pos[1] - py)

                vx = self.w * vx
                vy = self.w * vy

                dx = 1 if vx > 0.5 else (-1 if vx < -0.5 else 0)
                dy = 1 if vy > 0.5 else (-1 if vy < -0.5 else 0)

                best_neighbor = None
                best_score = -float('inf')
                for nx, ny in neighbors:
                    dir_score = dx * (nx - px) + dy * (ny - py)
                    end_dist = abs(nx - END[0]) + abs(ny - END[1])
                    score = dir_score - 0.1 * end_dist
                    if score > best_score:
                        best_score = score
                        best_neighbor = (nx, ny)

                if best_neighbor:
                    new_position = particle['position'] + [best_neighbor]
                    particle['velocity'] = [best_neighbor[0] - px, best_neighbor[1] - py]
                    score = self.evaluate_path(new_position)
                    if score < particle['best_score']:
                        particle['best_position'] = new_position.copy()
                        particle['best_score'] = score
                    if score < self.best_length:
                        self.best_length = score
                        self.best_path = new_position.copy()
                    particle['position'] = new_position

        return self.best_path


class SVM:
    def __init__(self, maze, C=1.0, kernel='rbf', gamma='scale', max_iter=1000):
        self.maze = maze
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.max_iter = max_iter
        self.best_path = None
        self.model = None
        self.features = []
        self.labels = []
        self.is_trained = False  # Track training status

    def extract_features(self, pos):
        x, y = pos
        features = [
            x, y,
            abs(x - END[0]),
            abs(y - END[1]),
            self.maze[y][x]
        ]
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                features.append(self.maze[ny][nx])
                distance = abs(nx - END[0]) + abs(ny - END[1])
                # Replace infinity with a large finite value
                features.append(min(distance, GRID_WIDTH + GRID_HEIGHT))
            else:
                features.append(1)  # Wall
                features.append(GRID_WIDTH + GRID_HEIGHT)  # Max possible distance
        return np.array(features).reshape(1, -1)

    def generate_training_data(self):
        queue = deque()
        queue.append((START, [START]))
        visited = set()
        good_paths = []

        # Generate some valid paths
        for _ in range(5):  # Try to find 5 good paths
            current_path = [START]
            current_pos = START
            visited = set([START])

            for _ in range(GRID_WIDTH * GRID_HEIGHT):
                neighbors = [n for n in self.get_neighbors(current_pos) if n not in visited]
                if not neighbors:
                    break
                next_pos = random.choice(neighbors)
                current_path.append(next_pos)
                visited.add(next_pos)
                current_pos = next_pos
                if current_pos == END:
                    good_paths.append(current_path)
                    break

        # If no good paths found, create a simple default path
        if not good_paths:
            simple_path = [START]
            x, y = START
            while x < END[0]:
                x += 1
                simple_path.append((x, y))
            while y < END[1]:
                y += 1
                simple_path.append((x, y))
            good_paths.append(simple_path)

        # Generate positive samples
        for path in good_paths:
            for pos in path:
                features = self.extract_features(pos).flatten()
                self.features.append(features)
                self.labels.append(1)  # Positive class

        # Generate negative samples
        num_negative = len(self.features)
        for _ in range(num_negative):
            x, y = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
            while (x, y) == START or (x, y) == END or (x, y) in [p for path in good_paths for p in path]:
                x, y = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
            features = self.extract_features((x, y)).flatten()
            self.features.append(features)
            self.labels.append(-1)  # Negative class

    def get_neighbors(self, pos):
        x, y = pos
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and self.maze[ny][nx] == 0:
                neighbors.append((nx, ny))
        return neighbors

    def train(self):
        try:
            from sklearn import svm
            if not self.features:
                self.generate_training_data()

            X = np.array(self.features)
            y = np.array(self.labels)

            # Ensure no infinite or NaN values
            X = np.nan_to_num(X, nan=0.0, posinf=GRID_WIDTH + GRID_HEIGHT, neginf=0.0)

            self.model = svm.SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, max_iter=self.max_iter)
            self.model.fit(X, y)
            self.is_trained = True
        except ImportError:
            print("scikit-learn not available. Using fallback path.")
            self.is_trained = False
        except Exception as e:
            print(f"SVM training error: {e}")
            self.is_trained = False

    def predict_next_move(self, pos, path):
        if not self.is_trained:
            # If model isn't trained, use a simple heuristic
            neighbors = self.get_neighbors(pos)
            if not neighbors:
                return None
            # Move toward the end
            return min(neighbors, key=lambda p: abs(p[0] - END[0]) + abs(p[1] - END[1]))

        neighbors = self.get_neighbors(pos)
        if not neighbors:
            return None

        best_score = -float('inf')
        best_move = None
        for neighbor in neighbors:
            try:
                features = self.extract_features(neighbor)
                features = np.nan_to_num(features, nan=0.0, posinf=GRID_WIDTH + GRID_HEIGHT, neginf=0.0)
                score = self.model.decision_function(features)[0]
                if neighbor in path:
                    score -= 2.0  # Penalize revisiting
                if score > best_score:
                    best_score = score
                    best_move = neighbor
            except Exception as e:
                print(f"Prediction error: {e}")
                continue
        return best_move if best_move is not None else random.choice(neighbors)

    def run(self):
        self.train()
        path = [START]
        visited = set([START])

        for _ in range(GRID_WIDTH * GRID_HEIGHT * 2):  # Increased max steps
            current = path[-1]
            if current == END:
                break
            next_move = self.predict_next_move(current, path)
            if next_move is None or next_move in visited:
                break
            path.append(next_move)
            visited.add(next_move)

        # Ensure we have a valid path to the end
        if path[-1] != END:
            # Try to connect to the end
            last = path[-1]
            if last[0] < END[0]:
                path.append((last[0] + 1, last[1]))
            elif last[0] > END[0]:
                path.append((last[0] - 1, last[1]))
            elif last[1] < END[1]:
                path.append((last[0], last[1] + 1))
            elif last[1] > END[1]:
                path.append((last[0], last[1] - 1))
            path.append(END)

        self.best_path = path
        return self.best_path
class Perceptron:
    def __init__(self, maze, learning_rate=0.01, n_iterations=100):
        self.maze = maze
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.best_path = None

    def extract_features(self, pos):
        x, y = pos
        features = [
            1,
            x / GRID_WIDTH,
            y / GRID_HEIGHT,
            abs(x - END[0]) / GRID_WIDTH,
            abs(y - END[1]) / GRID_HEIGHT,
            1 if self.maze[y][x] == 1 else 0,
        ]
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                features.append(1 if self.maze[ny][nx] == 1 else 0)
                features.append((abs(nx - END[0]) + abs(ny - END[1])) / (GRID_WIDTH + GRID_HEIGHT))
            else:
                features.append(1)
                features.append(1.0)
        return features

    def generate_training_data(self):
        queue = deque()
        queue.append((START, [START]))
        visited = set()
        good_paths = []

        while queue and len(good_paths) < 5:  # Reduced from 10
            pos, path = queue.popleft()
            if pos in visited:
                continue
            visited.add(pos)

            if pos == END:
                good_paths.append(path)
                continue

            for neighbor in self.get_neighbors(pos):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

        X = []
        y = []

        for path in good_paths:
            for i in range(len(path) - 1):
                current = path[i]
                next_move = path[i + 1]
                X.append(self.extract_features(current))
                y.append(1)

        num_negative = len(y)
        for _ in range(num_negative):
            x, y_pos = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
            while (x, y_pos) == START or (x, y_pos) == END or (x, y_pos) in [p for path in good_paths for p in path]:
                x, y_pos = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
            X.append(self.extract_features((x, y_pos)))
            y.append(-1)

        return X, y

    def get_neighbors(self, pos):
        x, y = pos
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and self.maze[ny][nx] == 0:
                neighbors.append((nx, ny))
        return neighbors

    def train(self, X, y):
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.n_iterations):
            for xi, target in zip(X, y):
                activation = self.bias
                for i in range(n_features):
                    activation += self.weights[i] * xi[i]
                prediction = 1 if activation >= 0 else -1
                update = self.learning_rate * (target - prediction)
                self.bias += update
                for i in range(n_features):
                    self.weights[i] += update * xi[i]

    def predict(self, features):
        activation = self.bias
        for i in range(len(features)):
            activation += self.weights[i] * features[i]
        return 1 if activation >= 0 else -1

    def predict_next_move(self, pos, path):
        neighbors = self.get_neighbors(pos)
        if not neighbors:
            return None

        best_score = -float('inf')
        best_move = None
        current_features = self.extract_features(pos)

        for neighbor in neighbors:
            features = self.extract_features(neighbor)
            confidence = self.bias
            for i in range(len(features)):
                confidence += self.weights[i] * features[i]
            if neighbor in path:
                confidence -= 2.0
            if confidence > best_score:
                best_score = confidence
                best_move = neighbor

        return best_move

    def run(self):
        X, y = self.generate_training_data()
        self.train(X, y)

        path = [START]
        visited = set([START])

        for _ in range(GRID_WIDTH * GRID_HEIGHT):
            current = path[-1]
            if current == END:
                break
            next_move = self.predict_next_move(current, path)
            if next_move is None or next_move in visited:
                break
            path.append(next_move)
            visited.add(next_move)

        self.best_path = path if path[-1] == END else [START, END]
        return self.best_path


class Revolutionary:
    def __init__(self, maze, population_size=20, mutation_rate=0.1, revolution_rate=0.05):
        self.maze = maze
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.revolution_rate = revolution_rate
        self.best_path = None
        self.generation = 0
        self.consecutive_no_improvement = 0
        self.population = []

    def get_neighbors(self, pos):
        x, y = pos
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and self.maze[ny][nx] == 0:
                neighbors.append((nx, ny))
        return neighbors

    def initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            path = [START]
            current = START
            visited = set([START])

            for _ in range(random.randint(1, GRID_WIDTH + GRID_HEIGHT)):
                neighbors = [n for n in self.get_neighbors(current) if n not in visited]
                if not neighbors:
                    break
                next_pos = random.choice(neighbors)
                path.append(next_pos)
                visited.add(next_pos)
                current = next_pos
                if current == END:
                    break

            self.population.append(path)

    def evaluate_path(self, path):
        if not path or path[-1] != END:
            return 0

        score = 1.0 / len(path)
        unique_cells = len(set(path))
        score += 0.1 * (unique_cells / len(path))

        wall_penalty = 0
        for x, y in path:
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and self.maze[ny][nx] == 1:
                    wall_penalty += 0.01
        score -= wall_penalty

        return score

    def selection(self):
        tournament_size = 3
        parents = []
        for _ in range(self.population_size):
            contestants = random.sample(self.population, tournament_size)
            winner = max(contestants, key=lambda p: self.evaluate_path(p))
            parents.append(winner.copy())
        return parents

    def crossover(self, parent1, parent2):
        common_points = [pos for pos in parent1 if pos in parent2 and pos != START and pos != END]
        if not common_points:
            return parent1 if random.random() < 0.5 else parent2

        crossover_point = random.choice(common_points)
        idx1 = parent1.index(crossover_point)
        idx2 = parent2.index(crossover_point)
        return parent1[:idx1] + parent2[idx2:]

    def mutate(self, path):
        if len(path) < 2 or random.random() > self.mutation_rate:
            return path

        mutation_type = random.randint(0, 3)
        if mutation_type == 0:  # Add random segment
            insert_point = random.randint(1, len(path) - 1)
            current = path[insert_point - 1]
            new_segment = [current]
            for _ in range(random.randint(1, 3)):
                neighbors = self.get_neighbors(current)
                if not neighbors:
                    break
                current = random.choice(neighbors)
                new_segment.append(current)
                if current == END:
                    break
            return path[:insert_point] + new_segment + path[insert_point:]

        elif mutation_type == 1:  # Remove random segment
            start = random.randint(1, len(path) - 2)
            end = random.randint(start + 1, len(path) - 1)
            return path[:start] + path[end:]

        elif mutation_type == 2:  # Random walk from a point
            point = random.randint(0, len(path) - 1)
            current = path[point]
            new_path = path[:point + 1]
            for _ in range(random.randint(1, 5)):
                neighbors = self.get_neighbors(current)
                if not neighbors:
                    break
                current = random.choice(neighbors)
                new_path.append(current)
                if current == END:
                    break
            return new_path

        else:  # Local optimization
            for i in range(1, len(path) - 1):
                if random.random() < 0.3:
                    neighbors = self.get_neighbors(path[i - 1])
                    if neighbors:
                        path[i] = random.choice(neighbors)
            return path

    def revolution(self):
        num_revolutionaries = int(self.revolution_rate * self.population_size)
        for _ in range(num_revolutionaries):
            idx = random.randint(0, self.population_size - 1)
            self.population[idx] = self.create_revolutionary_path()

    def create_revolutionary_path(self):
        path_type = random.randint(0, 3)
        if path_type == 0:  # Straight line towards end
            path = [START]
            current = START
            while current != END:
                dx = 1 if END[0] > current[0] else (-1 if END[0] < current[0] else 0)
                dy = 1 if END[1] > current[1] else (-1 if END[1] < current[1] else 0)
                new_pos = (current[0] + dx, current[1] + dy)
                if new_pos[0] < 0 or new_pos[0] >= GRID_WIDTH or new_pos[1] < 0 or new_pos[1] >= GRID_HEIGHT:
                    new_pos = current
                elif self.maze[new_pos[1]][new_pos[0]] == 0:
                    path.append(new_pos)
                    current = new_pos
                    continue
                directions = [(dx, dy), (dx, 0), (0, dy)]
                random.shuffle(directions)
                for d in directions:
                    new_pos = (current[0] + d[0], current[1] + d[1])
                    if (0 <= new_pos[0] < GRID_WIDTH and 0 <= new_pos[1] < GRID_HEIGHT and
                            self.maze[new_pos[1]][new_pos[0]] == 0 and new_pos not in path):
                        path.append(new_pos)
                        current = new_pos
                        break
                else:
                    break
            return path

        elif path_type == 1:  # Spiral pattern
            path = [START]
            current = START
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            dir_idx = 0
            step_size = 1
            steps_taken = 0
            for _ in range(GRID_WIDTH * GRID_HEIGHT):
                dx, dy = directions[dir_idx]
                new_pos = (current[0] + dx, current[1] + dy)
                if (0 <= new_pos[0] < GRID_WIDTH and 0 <= new_pos[1] < GRID_HEIGHT and
                        self.maze[new_pos[1]][new_pos[0]] == 0 and new_pos not in path):
                    path.append(new_pos)
                    current = new_pos
                    if current == END:
                        break
                    steps_taken += 1
                    if steps_taken >= step_size:
                        dir_idx = (dir_idx + 1) % 4
                        if dir_idx % 2 == 0:
                            step_size += 1
                        steps_taken = 0
                else:
                    dir_idx = (dir_idx + 1) % 4
                    steps_taken = 0
            return path

        else:  # Random with momentum
            path = [START]
            current = START
            momentum = [random.choice([-1, 0, 1]), random.choice([-1, 0, 1])]
            for _ in range(GRID_WIDTH + GRID_HEIGHT):
                if random.random() < 0.7 and any(momentum):
                    new_pos = (current[0] + momentum[0], current[1] + momentum[1])
                    if (0 <= new_pos[0] < GRID_WIDTH and 0 <= new_pos[1] < GRID_HEIGHT and
                            self.maze[new_pos[1]][new_pos[0]] == 0 and new_pos not in path):
                        path.append(new_pos)
                        current = new_pos
                        if current == END:
                            break
                        continue
                neighbors = [n for n in self.get_neighbors(current) if n not in path]
                if neighbors:
                    next_pos = random.choice(neighbors)
                    path.append(next_pos)
                    current = next_pos
                    momentum = [next_pos[0] - current[0], next_pos[1] - current[1]]
                    if current == END:
                        break
                else:
                    break
            return path

    def run(self, max_generations=30):
        self.initialize_population()
        self.best_path = max(self.population, key=lambda p: self.evaluate_path(p), default=None)

        for _ in range(max_generations):
            parents = self.selection()
            new_population = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1 = self.crossover(parents[i], parents[i + 1])
                    child2 = self.crossover(parents[i + 1], parents[i])
                    new_population.extend([child1, child2])
                else:
                    new_population.append(parents[i])

            self.population = [self.mutate(p) for p in new_population]

            if self.generation % 10 == 0 or self.consecutive_no_improvement > 5:
                self.revolution()

            if self.best_path:
                worst_idx = min(range(len(self.population)),
                                key=lambda i: self.evaluate_path(self.population[i]))
                self.population[worst_idx] = self.best_path.copy()

            current_best = max(self.population, key=lambda p: self.evaluate_path(p))
            if self.evaluate_path(current_best) > self.evaluate_path(self.best_path):
                self.best_path = current_best.copy()
                self.consecutive_no_improvement = 0
            else:
                self.consecutive_no_improvement += 1

            self.generation += 1

            if self.best_path and self.best_path[-1] == END and self.consecutive_no_improvement > 10:
                break

        self.best_path = self.best_path if (self.best_path and self.best_path[-1] == END) else [START, END]
        return self.best_path


# === Game Functions ===
def is_near_path(pos, path, tolerance=1):
    x, y = pos
    for px, py in path:
        if abs(px - x) <= tolerance and abs(py - y) <= tolerance:
            return True
    return False


def draw_grid(screen, maze, player_pos, best_path=None, show_best=False):
    screen.fill(WHITE)
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if maze[y][x] == 1:
                pygame.draw.rect(screen, BLACK, rect)
            else:
                pygame.draw.rect(screen, GRAY, rect, 1)

    if best_path and show_best:
        for (x, y) in best_path:
            pygame.draw.rect(screen, ORANGE, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.draw.rect(screen, GREEN, (player_pos[0] * CELL_SIZE, player_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(screen, RED, (END[0] * CELL_SIZE, END[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))


def show_win_screen(screen, font, elapsed_time, algorithm):
    overlay = pygame.Surface((WIDTH, HEIGHT - 130), pygame.SRCALPHA)
    overlay.fill((255, 255, 255, 180))
    screen.blit(overlay, (0, 0))

    win_surf = font.render("🎉 You reached the goal!", True, GREEN)
    win_rect = win_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 30))
    screen.blit(win_surf, win_rect)

    time_surf = font.render(f"Time: {elapsed_time:.2f} seconds", True, BLACK)
    time_rect = time_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 10))
    screen.blit(time_surf, time_rect)

    algo_surf = font.render(f"Algorithm: {algorithm}", True, BLUE)
    algo_rect = algo_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
    screen.blit(algo_surf, algo_rect)

    restart_btn = Button(WIDTH // 2 - 60, HEIGHT // 2 + 90, 120, 40, "Play Again", None, draggable=False)
    restart_btn.draw(screen, font)

    pygame.display.flip()
    return restart_btn


def algorithm_menu(screen, font):
    algorithms = ["ACO", "PSO", "SVM", "Perceptron", "Revolutionary"]
    buttons = []
    btn_width = 200
    btn_height = 50
    btn_spacing = 20
    total_height = len(algorithms) * btn_height + (len(algorithms) - 1) * btn_spacing
    start_y = (HEIGHT - total_height) // 2

    for i, algo in enumerate(algorithms):
        btn = Button(WIDTH // 2 - btn_width // 2,
                     start_y + i * (btn_height + btn_spacing),
                     btn_width, btn_height, algo)
        buttons.append(btn)

    title = font.render("Select Pathfinding Algorithm", True, BLUE)
    title_rect = title.get_rect(center=(WIDTH // 2, start_y - 50))

    running = True
    selected_algorithm = None

    while running:
        screen.fill(WHITE)
        screen.blit(title, title_rect)

        for btn in buttons:
            btn.draw(screen, font)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                selected_algorithm = None
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
                selected_algorithm = None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for i, btn in enumerate(buttons):
                    if btn.is_over(event.pos):
                        selected_algorithm = algorithms[i]
                        running = False

    return selected_algorithm


def initialize_game(algorithm):
    maze = generate_maze()
    if algorithm == "ACO":
        algo_obj = ACO(maze)
    elif algorithm == "PSO":
        algo_obj = PSO(maze)
    elif algorithm == "SVM":
        algo_obj = SVM(maze)
    elif algorithm == "Perceptron":
        algo_obj = Perceptron(maze)
    elif algorithm == "Revolutionary":
        algo_obj = Revolutionary(maze)
    else:
        algo_obj = ACO(maze)  # Default to ACO

    algo_obj.run()
    return maze, algo_obj, algorithm


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Smart Maze Game - AI Algorithms")
    font = pygame.font.SysFont(None, 28)
    small_font = pygame.font.SysFont(None, 24)

    try:
        algorithm = algorithm_menu(screen, font)
        if algorithm is None:
            pygame.quit()
            return

        maze, algo_obj, algorithm = initialize_game(algorithm)
        control_bar = ControlBar(WIDTH)

        player_pos = list(START)
        clock = pygame.time.Clock()
        running = True
        show_best = False
        off_path_warning = False
        start_time = time.time()

        speed = 5
        move_delay = 1 / speed
        last_move_time = 0

        while running:
            current_time = time.time()
            elapsed_time = current_time - start_time

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_b:
                        show_best = not show_best

                result = control_bar.handle_event(event)
                if result:
                    if result["type"] == "speed":
                        speed = result["value"]
                        move_delay = 1 / speed
                    elif result["type"] == "toggle_path":
                        show_best = not show_best
                    elif result["type"] == "restart":
                        maze, algo_obj, algorithm = initialize_game(algorithm)
                        player_pos = list(START)
                        start_time = time.time()
                    elif result["type"] == "algorithm":
                        algorithm = algorithm_menu(screen, font)
                        if algorithm is None:
                            running = False
                        else:
                            maze, algo_obj, algorithm = initialize_game(algorithm)
                            player_pos = list(START)
                            start_time = time.time()

            if current_time - last_move_time > move_delay:
                dx, dy = 0, 0
                if pygame.key.get_pressed()[pygame.K_UP]:
                    dy = -1
                elif pygame.key.get_pressed()[pygame.K_DOWN]:
                    dy = 1
                elif pygame.key.get_pressed()[pygame.K_LEFT]:
                    dx = -1
                elif pygame.key.get_pressed()[pygame.K_RIGHT]:
                    dx = 1

                if dx != 0 or dy != 0:
                    new_x = player_pos[0] + dx
                    new_y = player_pos[1] + dy
                    if 0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT:
                        if maze[new_y][new_x] == 0:
                            player_pos = [new_x, new_y]
                    last_move_time = current_time

            if algo_obj.best_path:
                if tuple(player_pos) == START or tuple(player_pos) == END:
                    off_path_warning = False
                elif not is_near_path(tuple(player_pos), algo_obj.best_path, tolerance=1):
                    off_path_warning = True
                else:
                    off_path_warning = False

            draw_grid(screen, maze, player_pos, algo_obj.best_path, show_best)
            control_bar.update_status(algorithm, elapsed_time, off_path_warning)
            control_bar.draw(screen)
            pygame.display.flip()

            if tuple(player_pos) == END:
                restart_btn = show_win_screen(screen, font, elapsed_time, algorithm)
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            waiting = False
                            running = False
                        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                            waiting = False
                            running = False
                        elif event.type == pygame.MOUSEBUTTONDOWN:
                            if restart_btn.is_over(event.pos):
                                maze, algo_obj, algorithm = initialize_game(algorithm)
                                player_pos = list(START)
                                start_time = time.time()
                                show_best = False
                                off_path_warning = False
                                waiting = False

            clock.tick(60)

    except Exception as e:
        print(f"Game error: {e}")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()