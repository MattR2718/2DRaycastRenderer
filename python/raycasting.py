import numpy as np
import math

import pygame
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    K_SPACE,
    K_RETURN,
    KEYDOWN,
    QUIT,
)

#Initialise pygame
pygame.init()

#Set constants
WIDTH = 800
HEIGHT = 600
SCREEN_WIDTH = 2 * WIDTH
SCREEN_HEIGHT = HEIGHT
BLOCKSIZE = 50
CONE_ANGLE = 90
DEG_STEP = 1

#Load pre made map
#Add path
LOAD_MAP = "map.txt"
#LOAD_MAP = None

#Set up drawing window
screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
pygame.display.set_caption("Raycasting")

#canvas = pygame.Surface((800, 600))

# Camera rectangles for sections of  the canvas
FlatView = pygame.Rect(0,0,WIDTH,HEIGHT)
Render = pygame.Rect(WIDTH,0,WIDTH,HEIGHT)

#Create subsurfaces to seperate 2D view and 3D render
flat_sub = screen.subsurface(FlatView)
render_sub = screen.subsurface(Render)


#Get font for fps
font = pygame.font.SysFont("Arial", 18)

#Get clock to set fps
fps_clock = pygame.time.Clock()
FPS = 60

#Create a class for the player
class Player(pygame.sprite.Sprite):
	def __init__(self, x=1, y=1):
		super(Player, self).__init__()
		self.x = x
		self.y = y
		
		self.surf = pygame.Surface((BLOCKSIZE - 4, BLOCKSIZE - 4))
		self.surf.fill((0, 200, 0))
		self.rect = self.surf.get_rect(center=((self.x * BLOCKSIZE) + BLOCKSIZE//2, (self.y * BLOCKSIZE) + BLOCKSIZE//2))
		
	def update(self, pressed_keys):
		#Move player based off of keys pressed
		if pressed_keys[K_UP]:
			self.rect.move_ip(0, -10)
		if pressed_keys[K_DOWN]:
			self.rect.move_ip(0, 10)
		if pressed_keys[K_LEFT]:
			self.rect.move_ip(-10, 0)
		if pressed_keys[K_RIGHT]:
			self.rect.move_ip(10, 0)

		#Keep player on screen
		if self.rect.left < 0:
			self.rect.left = 0
		if self.rect.right > WIDTH:
			self.rect.right = WIDTH
		if self.rect.top < 0:
			self.rect.top = 0
		if self.rect.bottom > HEIGHT:
			self.rect.bottom = HEIGHT

#Draws a grid on the screen based on BLOCKSIZE, WIDTH and HEIGHT
def draw_grid():
    for x in range(WIDTH // BLOCKSIZE):
        for y in range(HEIGHT // BLOCKSIZE):
            rect = pygame.Rect(x*BLOCKSIZE, y*BLOCKSIZE, BLOCKSIZE, BLOCKSIZE)
            pygame.draw.rect(flat_sub, (0,0,0), rect, 1)

#Updates fps counter on screen
def update_fps():
	#Get fps from clock
	fps = str(int(fps_clock.get_fps()))
	#Set text for fps
	fps_text = font.render(fps, 1, pygame.Color("coral"))
	return fps_text

#Sets a border around the edge of the screen
def border():
	for x in range(WIDTH//BLOCKSIZE):
		grid[x][0] = 1
		grid[x][HEIGHT//BLOCKSIZE - 1] = 1
	for y in range(HEIGHT//BLOCKSIZE):
		grid[0][y] = 1
		grid[WIDTH//BLOCKSIZE - 1][y] = 1

#Gets slopes for range of rays based off of a cone centered around mouse position
def get_slope():
	x1, y1 = player.rect.center
	x2, y2 = pygame.mouse.get_pos()
	slopes = []
	points = []
	#Rotates mouse point to create rays rotated i degrees anti_clockwise around player
	for i in range(int(360 - CONE_ANGLE / 2), (361 - DEG_STEP), DEG_STEP):
		points.append(rotate((x1, y1), (x2, y2), i * (math.pi/180)))
	#Rotates mouse point to create rays rotated i degrees clockwise around player
	for i in range(0, int(CONE_ANGLE / 2 + 1), DEG_STEP):
		points.append(rotate((x1, y1), (x2, y2), i * (math.pi/180)))
	
	#For each point calculate the slope which will be used for collision detection
	#m = (y2 - y1) / (x2 - x1)
	for point in points:
		if not x1 == point[0]:
			#If m = 0 then there will be divide by 0 errors
			if (point[1] - y1) / (point[0] - x1) == 0:
				slopes.append(0.001)
			else:
				slopes.append((point[1] - y1) / (point[0] - x1))
		#If points directly on y axis, slope gets very large so set to 300
		elif point[1] > y1:
			slopes.append(300)
		else:
			slopes.append(-300)
	#print(points, slopes)
	return slopes

def rotate(origin, point, angle):
    #Rotate a point counterclockwise by a given angle around a given origin.
    #The angle should be given in radians.
    
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return (qx, qy)
		
#Check if two points are on the same side of the payer as the mouse
#Used to remove all collisions on opposite side of player to mouse
def same_side(point):
	x1, y1 = pygame.mouse.get_pos()
	x2, y2 = point
	px, py = player.rect.center
	
	#y = mx + c
	
	#m = y2-y1/x2-x1
	if not x1 == px:
		if (y1 - py) / (x1 - px) == 0:
			m = 0.001
		else:
			m = (y1 - py) / (x1 - px)
	elif y1 > py:
		m = 300
	else:
		m = -300
	
	#Negative reciprocal for line perpendicular to mouse line
	m = -1/m
	
	#c = y - mx
	c = py - m*px
	
	#Draw perpendicular mouse line
	pygame.draw.line(flat_sub, (0,0,255), (int((-c)/m), 0), (int((600 - c) / m), 600), 3)
	
	#Check if both points are on same side
	if (y1 >= m * x1 + c and y2 >= m * x2 + c) or (y1 <= m * x1 + c and y2 <= m *x2 + c):
		return True #both on same side 
	else:
		return False #points on different side

#Check collisions on all wall blocks in grid
def check_collisions(x, y, m, collisions):
	x1, y1 = player.rect.center
	x2, y2 = pygame.mouse.get_pos()

	#Work out y intercept of ray
	c = y1 - x1 * m
	
	#Define borders of wall
	#Left x, right x, top y, bottom y
	lx = x * BLOCKSIZE
	rx = x * BLOCKSIZE + BLOCKSIZE
	ty = y * BLOCKSIZE
	by = y * BLOCKSIZE + BLOCKSIZE
	
	#Check x axis collisions
	ly = m * lx + c
	ry = m * rx + c
	if m > 0:
		if (ly <= ty and ry >= ty):
			itx = (ty - c) // m
			if same_side((int(itx), ty)):
				collisions.append((int(itx), ty))
		if (ly <= by and ry >= by):
			ibx = (by - c) // m
			if same_side((int(ibx), by)):
				collisions.append((int(ibx), by))
	#For negative m, flip comparison operator
	elif m < 0:
		if (ly >= ty and ry <= ty):
			itx = (ty - c) // m
			if same_side((int(itx), ty)):
				collisions.append((int(itx), ty))
		if (ly >= by and ry <= by):
			ibx = (by - c) // m
			if same_side((int(ibx), by)):
				collisions.append((int(ibx), by))
	else:
		print("ERROR WITH X AXIS COLLISIONS M SIGN")
	
	#Check y axis collisions
	tx = (ty - c) / m
	bx = (by - c) / m
	if m > 0:
		if (tx <= lx and bx >= lx):
			ily = m * lx + c
			if same_side((lx, int(ily))):
				collisions.append((lx, int(ily)))
		if (tx <= rx and bx >= rx):
			iry = m * rx + c
			if same_side((rx, int(iry))):
				collisions.append((rx, int(iry)))
	elif m < 0:
		if (tx >= lx and bx <= lx):
			ily = m * lx + c
			if same_side((lx, int(ily))):
				collisions.append((lx, int(ily)))
		if (tx >= rx and bx <= rx):
			iry = m * rx + c
			if same_side((rx, int(iry))):
				collisions.append((rx, int(iry)))
	else:
		print("ERROR WITH Y AXIS COLLISIONS M SIGN")
		
#Find closest collision between block and ray
def find_closest_collision(collisions):
	#print(collisions)
	x1, y1 = player.rect.center
	short_dist = 100000
	shortest = (0,0)
	for hits in collisions:
		#Check distance between collision and player
		dist = math.sqrt((hits[0] - x1) ** 2 + (hits[1] - y1) ** 2)
		if dist < short_dist:
			short_dist = dist
			shortest = hits
	#Draw line between player and closest collision
	pygame.draw.line(flat_sub, (255,255,0), (player.rect.center[0], player.rect.center[1]), (shortest[0], shortest[1]), 3)
	return short_dist

#Draw a circle at every collision
def draw_collisions(collisions):
	for hit in collisions:
		pygame.draw.circle(flat_sub, (255,0,0), (hit[0], hit[1]), 5)
		
#Casts rays out from player
def raycast():
	#Gets an array of slopes in a cone of vision from player
	slopes = get_slope()
	closest_collisions = []
	for slope in slopes:
		collisions = []
		mx, my = pygame.mouse.get_pos()
		#Check collisions of every m in slopes
		for x in range(grid.shape[0]):
			for y in range(grid.shape[1]):
				if grid[x][y] == 1:
					check_collisions(x, y, slope, collisions)
					#pass
		
		#draw_collisions(collisions)
		closest_collisions.append(find_closest_collision(collisions))
	return closest_collisions

#Colours floor and ceiling of render window
def room_floors():
	#Ceiling
	pygame.draw.rect(render_sub, (50,50,50), (0,0, WIDTH, int(HEIGHT/2)))
	#Floor
	pygame.draw.rect(render_sub, (0,0,150), (0,int(HEIGHT/2), WIDTH, int(HEIGHT/2)))

#Maps value from one range to another
def map_values(value, leftMin, leftMax, rightMin, rightMax):
	# Figure out how 'wide' each range is
	leftSpan = leftMax - leftMin
	rightSpan = rightMax - rightMin

	# Convert the left range into a 0-1 range (float)
	valueScaled = float(value - leftMin) / float(leftSpan)

	# Convert the 0-1 range into a value in the right range.
	return rightMin + (valueScaled * rightSpan)

#Renders room by drawing rectange of proportionate size to collision distance
def render_room(closest_collisions):
	#Get center line of window
	center_y = int(HEIGHT / 2)
	for idx, value in enumerate(closest_collisions):
		#Map values
		mapped_value = map_values(value, 0, WIDTH, int((2/3) * HEIGHT), 50)	
		
		#Get coordinate of top left corner
		top_x = int(idx*int((WIDTH / (CONE_ANGLE / DEG_STEP))))
		top_y = int(center_y - mapped_value / 2)
		#Get width and height of rectangle
		width = int((WIDTH / (CONE_ANGLE / DEG_STEP)))
		height = int(mapped_value)
		#Map colour based off of distance between 'wall' and player
		#Simulates light dimming the further an object is away 
		map_colour = 255 - int(map_values(mapped_value, 0, 400, 240, 0))
		colour = (0, map_colour, 50)
		pygame.draw.rect(render_sub, colour, (top_x, top_y, width, height))
	
#Create a player object
player = Player()

#If no map is loaded create an empty map with a border
#Otherwise load map from text file
grid = np.zeros(((WIDTH//BLOCKSIZE), (HEIGHT//BLOCKSIZE)))
if LOAD_MAP == None:
	border()
	grid[player.x][player.y] = 2
else:
	grid = np.loadtxt(LOAD_MAP, dtype=int)
	#print(grid)

running = True
place_objects = True
while running:
	# Did the user click the window close button?
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False
		#If key has been pressed
		if event.type == KEYDOWN:
			#If space bar is pressed, stop allowing objects to be placed and reset border
			if event.key == K_SPACE:
				place_objects = False
				border()
				print("SPACEBAR")
			#If enter is placed reset player position
			if event.key == K_RETURN:
				print("ENTER")
				player.__init__()
			
		#If can place objects check if mouse is pressed and update grid
		if place_objects:
			#If mouse button is down then place object in grid
			if event.type == pygame.MOUSEBUTTONDOWN:
				mouse_pos = pygame.mouse.get_pos()
				print("Mouse Pressed at: ", mouse_pos)
				if grid[mouse_pos[0]//BLOCKSIZE][mouse_pos[1]//BLOCKSIZE] == 1:
					grid[mouse_pos[0]//BLOCKSIZE][mouse_pos[1]//BLOCKSIZE] = 0
					
				elif grid[mouse_pos[0]//BLOCKSIZE][mouse_pos[1]//BLOCKSIZE] == 0:
					grid[mouse_pos[0]//BLOCKSIZE][mouse_pos[1]//BLOCKSIZE] = 1
					

	#Fill screen backgrounds with colour
	flat_sub.fill((255, 255, 255))
	render_sub.fill((0, 0, 255))

	#Draw Grid
	draw_grid()


	#Draw objects from grid on raycast screen
	for x in range(grid.shape[0]):
		for y in range(grid.shape[1]):
			if grid[x][y] == 1:
				pygame.draw.rect(flat_sub, (0,0,0), (x*BLOCKSIZE, y*BLOCKSIZE, BLOCKSIZE, BLOCKSIZE))
			elif grid[x][y] == 2:
				pygame.draw.rect(flat_sub, (255,255,255), (x*BLOCKSIZE, y*BLOCKSIZE, BLOCKSIZE, BLOCKSIZE))
	
	if not place_objects:
		#Get set of keys pressed and check for user input
		pressed_keys = pygame.key.get_pressed()
		#Player movement based on pressed keys
		player.update(pressed_keys)
		#Draw player on flat_sub
		flat_sub.blit(player.surf, player.rect)
		#Call raycast to draw rays and calculate collisions
		closest_collisions = raycast()
		#Draw floors and ceiling in render window
		room_floors()
		#Render map based off of closest collisions
		render_room(closest_collisions)
		
			
	#Draw fps on screen
	flat_sub.blit(update_fps(), (10,0))
	
	#pygame.draw.line(flat_sub, (255,0,0), (0,0), (WIDTH, HEIGHT), 10)
	
	#Update display
	pygame.display.flip()
	
	fps_clock.tick(FPS)

pygame.quit()
