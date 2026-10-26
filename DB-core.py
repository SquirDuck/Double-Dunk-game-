import pygame
import sys
import math

pygame.init()
WIDTH, HEIGHT = 640, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Double Dunk Prototype")
clock = pygame.time.Clock()

# --- Colors ---
COURT_COLOR = (205, 133, 63)
LINE_COLOR = (255, 255, 255)
PLAYER_COLOR = (30, 144, 255)
DEFENDER_COLOR = (220, 20, 60)
BALL_COLOR = (255, 140, 0)
HOOP_COLOR = (255, 69, 0)

# --- Hoop Setup ---
HOOP_POS = pygame.Vector2(WIDTH // 2, 70)
HOOP_RADIUS = 14

class Ball:
    def __init__(self):
        self.pos = pygame.Vector2(WIDTH // 2, 200)
        self.vel = pygame.Vector2(0, 0)
        self.z = 0.0          # Height above floor
        self.z_vel = 0.0      # Vertical velocity
        self.gravity = -0.35
        self.holder = None
        self.in_flight = False

    def shoot(self, target):
        if self.holder:
            self.pos = pygame.Vector2(self.holder.pos)
            self.holder = None
            self.in_flight = True
            
            # 2D direction toward target
            direction = target - self.pos
            dist = direction.length()
            if dist > 0:
                direction = direction.normalize()
            
            travel_time = max(25, int(dist / 4.5))
            self.vel = (target - self.pos) / travel_time
            # Calculate upward impulse to land at z=0 near destination
            self.z = 10.0
            self.z_vel = -0.5 * self.gravity * travel_time

    def update(self):
        if self.holder:
            self.pos = self.holder.pos + pygame.Vector2(8, 0)
            self.z = 0.0
        elif self.in_flight:
            self.pos += self.vel
            self.z += self.z_vel
            self.z_vel += self.gravity
            
            # Hit the ground
            if self.z <= 0:
                self.z = 0
                self.in_flight = False
                self.vel = pygame.Vector2(0, 0)

    def draw(self, surface):
        # Draw shadow on floor
        shadow_radius = max(3, int(7 - self.z * 0.1))
        pygame.draw.circle(surface, (80, 50, 20), (int(self.pos.x), int(self.pos.y)), shadow_radius)
        # Draw elevated ball
        screen_y = int(self.pos.y - self.z)
        pygame.draw.circle(surface, BALL_COLOR, (int(self.pos.x), screen_y), 7)


class Player:
    def __init__(self, x, y, color, is_player=True):
        self.pos = pygame.Vector2(x, y)
        self.speed = 3.5
        self.color = color
        self.is_player = is_player
        self.radius = 12

    def update(self, keys=None, ball=None):
        if self.is_player and keys:
            move = pygame.Vector2(0, 0)
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                move.x -= 1
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                move.x += 1
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                move.y -= 1
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                move.y += 1
            
            if move.length() > 0:
                self.pos += move.normalize() * self.speed
        elif not self.is_player and ball:
            # Simple AI: Stay between ball and hoop
            target = (ball.pos + HOOP_POS) / 2
            diff = target - self.pos
            if diff.length() > 2:
                self.pos += diff.normalize() * 2.0

        # Boundary constraints
        self.pos.x = max(60, min(WIDTH - 60, self.pos.x))
        self.pos.y = max(50, min(HEIGHT - 40, self.pos.y))

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.pos.x), int(self.pos.y)), self.radius)


# --- Game Setup ---
player = Player(WIDTH // 2, 320, PLAYER_COLOR, is_player=True)
defender = Player(WIDTH // 2, 180, DEFENDER_COLOR, is_player=False)
ball = Ball()
ball.holder = player

score = 0
font = pygame.font.SysFont(None, 28)

# --- Main Game Loop ---
running = True
while running:
    keys = pygame.key.get_pressed()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and ball.holder == player:
                ball.shoot(HOOP_POS)

    # Updates
    player.update(keys=keys)
    defender.update(ball=ball)
    ball.update()

    # Rebound / Pick up ball if loose
    if not ball.holder and not ball.in_flight:
        if (player.pos - ball.pos).length() < (player.radius + 7):
            ball.holder = player
        elif (defender.pos - ball.pos).length() < (defender.radius + 7):
            ball.holder = defender

    # Scoring Check
    if ball.in_flight and (ball.pos - HOOP_POS).length() < HOOP_RADIUS and ball.z < 15:
        score += 2
        ball.in_flight = False
        ball.holder = defender  # Turnover on basket
        ball.pos = pygame.Vector2(WIDTH // 2, 100)

    # --- Render ---
    screen.fill(COURT_COLOR)
    
    # Half-court markings
    pygame.draw.rect(screen, LINE_COLOR, (40, 30, WIDTH - 80, HEIGHT - 60), 3)
    pygame.draw.circle(screen, LINE_COLOR, (WIDTH // 2, HEIGHT - 30), 70, 3) # Free-throw/Key line
    pygame.draw.arc(screen, LINE_COLOR, (WIDTH // 2 - 130, 40, 260, 240), 0, math.pi, 3) # 3-pt line
    
    # Backboard & Hoop
    pygame.draw.line(screen, (200, 200, 200), (WIDTH // 2 - 30, 50), (WIDTH // 2 + 30, 50), 6)
    pygame.draw.circle(screen, HOOP_COLOR, (int(HOOP_POS.x), int(HOOP_POS.y)), HOOP_RADIUS, 3)

    # Entities
    player.draw(screen)
    defender.draw(screen)
    ball.draw(screen)

    # UI
    score_txt = font.render(f"Score: {score}  |  [Arrow Keys/WASD] Move  [SPACE] Shoot", True, (255, 255, 255))
    screen.blit(score_txt, (50, HEIGHT - 25))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
